"""Advanced Retinal Field of View (FOV) Auto-Cropping & Illumination Normalization.

Implements precision circular/elliptical FOV mask extraction, Graham's local average subtraction,
and coordinate tracking for seamless un-cropping / re-mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class FOVResult:
    """Dataclass storing FOV extraction outputs and spatial transformation metadata.

    Attributes:
        cropped_image: RGB image cropped to the retina ROI (H_crop, W_crop, 3), uint8.
        fov_mask: Binary uint8 mask array of shape (H_crop, W_crop) with 255 inside retina, 0 outside.
        crop_bbox: Tuple of (x1, y1, w, h) bounding box in original image coordinates.
        orig_shape: Tuple of (H_orig, W_orig) dimensions of the raw input image.
    """

    cropped_image: np.ndarray
    fov_mask: np.ndarray
    crop_bbox: tuple[int, int, int, int]
    orig_shape: tuple[int, int]


class FundusFOVExtractor:
    """Precision Field of View Extractor and Preprocessor for Fundus Photography."""

    def __init__(
        self,
        black_threshold: int = 12,
        margin_pct: float = 0.02,
        circle_fit: bool = True,
    ) -> None:
        """Initialize FundusFOVExtractor.

        Args:
            black_threshold: Pixel intensity threshold below which pixels are treated as background.
            margin_pct: Extra padding percentage preserved around the detected retina circle.
            circle_fit: Whether to fit a smooth geometric circle around retina contour to remove edge noise.
        """
        self.black_threshold = black_threshold
        self.margin_pct = margin_pct
        self.circle_fit = circle_fit

    def extract_fov_mask(
        self,
        image_rgb: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Extract a smooth circular/elliptical binary FOV mask and bounding box.

        Args:
            image_rgb: Input RGB fundus photograph (H, W, 3), uint8.

        Returns:
            Tuple of (Binary mask array (H, W) uint8 [0, 255], Bounding box tuple (x, y, w, h)).
        """
        if image_rgb.dtype != np.uint8:
            img_uint8 = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
        else:
            img_uint8 = image_rgb

        h_orig, w_orig = img_uint8.shape[:2]

        # Use Green Channel (index 1) for optimal vessel and retina boundary contrast
        green_ch = img_uint8[:, :, 1]
        _, binary = cv2.threshold(green_ch, self.black_threshold, 255, cv2.THRESH_BINARY)

        # Morphological closing with a large kernel (31x31) to fill large dark hemorrhages and vessel shadows
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            full_mask = np.ones((h_orig, w_orig), dtype=np.uint8) * 255
            return full_mask, (0, 0, w_orig, h_orig)

        raw_cnt = max(contours, key=cv2.contourArea)
        # Convex hull guarantees large dark interior lesions do not break the retina boundary
        largest_cnt = cv2.convexHull(raw_cnt)
        x, y, w, h = cv2.boundingRect(largest_cnt)

        # Apply margin padding
        margin_x = int(w * self.margin_pct)
        margin_y = int(h * self.margin_pct)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_orig, x + w + margin_x)
        y2 = min(h_orig, y + h + margin_y)

        crop_w = x2 - x1
        crop_h = y2 - y1

        mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

        if self.circle_fit:
            # Fit minimal enclosing circle with a strict inner boundary to prevent mask spill onto black background
            (cx, cy), radius = cv2.minEnclosingCircle(largest_cnt)
            strict_radius = max(1.0, radius * (1.0 - self.margin_pct * 0.5))
            cv2.circle(mask, (int(cx), int(cy)), int(strict_radius), 255, -1)
        else:
            cv2.drawContours(mask, [largest_cnt], -1, 255, thickness=cv2.FILLED)

        return mask, (x1, y1, crop_w, crop_h)

    @staticmethod
    def apply_grahams_preprocessing(
        image_rgb: np.ndarray,
        sigma: float = 10.0,
    ) -> np.ndarray:
        """Apply Graham's method (local average subtraction) for uniform retinal illumination.

        formula: I_new = alpha * I + beta * GaussianBlur(I, sigma) + gamma

        Args:
            image_rgb: Input RGB image (H, W, 3), uint8.
            sigma: Gaussian blur standard deviation controlling local neighborhood size.

        Returns:
            Illumination-normalized RGB image (H, W, 3), uint8.
        """
        if image_rgb.dtype != np.uint8:
            img = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
        else:
            img = image_rgb

        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        # alpha=4.0, beta=-4.0, gamma=128 for classic Graham enhancement
        enhanced = cv2.addWeighted(img, 4.0, blurred, -4.0, 128)
        return enhanced

    def process(
        self,
        image_rgb: np.ndarray,
        apply_graham: bool = False,
        graham_sigma: float = 10.0,
    ) -> FOVResult:
        """Process fundus image: extract FOV mask, crop ROI, and optionally normalize lighting.

        Args:
            image_rgb: Input RGB fundus image (H, W, 3), uint8.
            apply_graham: Whether to apply Graham's local average subtraction.
            graham_sigma: Sigma for Graham's Gaussian blur.

        Returns:
            FOVResult dataclass containing cropped image, cropped mask, and crop bbox metadata.
        """
        if image_rgb.dtype != np.uint8:
            img = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
        else:
            img = image_rgb.copy()

        h_orig, w_orig = img.shape[:2]
        full_mask, (x, y, w, h) = self.extract_fov_mask(img)

        # Mask background to pure black (0, 0, 0)
        masked_img = cv2.bitwise_and(img, img, mask=full_mask)

        if apply_graham:
            masked_img = self.apply_grahams_preprocessing(masked_img, sigma=graham_sigma)
            # Re-apply mask after Graham to keep background pure black
            masked_img = cv2.bitwise_and(masked_img, masked_img, mask=full_mask)

        cropped_img = masked_img[y : y + h, x : x + w]
        cropped_mask = full_mask[y : y + h, x : x + w]

        return FOVResult(
            cropped_image=cropped_img,
            fov_mask=cropped_mask,
            crop_bbox=(x, y, w, h),
            orig_shape=(h_orig, w_orig),
        )
