"""Retinal Field of View (FOV) Auto-Cropping & ROI Masking Utilities.

Eliminates empty outer black borders and edge-contrast artifacts around fundus photographs,
forcing models and XAI attribution algorithms to evaluate exclusively inside the retina tissue.
"""

from __future__ import annotations

import cv2
import numpy as np


def crop_fundus_roi(
    image_rgb: np.ndarray,
    black_threshold: int = 15,
    margin_pct: float = 0.02,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Automatically crop uninformative outer black margins around the retina circle.

    Args:
        image_rgb: Input RGB fundus image (H, W, 3), uint8.
        black_threshold: Pixel intensity threshold below which pixels are treated as black background.
        margin_pct: Additional padding margin percentage to keep around the retina bounding box.

    Returns:
        Tuple of (Cropped RGB image, bounding box tuple (x, y, w, h)).
    """
    if image_rgb.dtype != np.uint8:
        img_uint8 = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
    else:
        img_uint8 = image_rgb.copy()

    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_uint8, (0, 0, img_uint8.shape[1], img_uint8.shape[0])

    # Find the largest non-black contour (the retina circle)
    largest_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_cnt)

    # Add margin
    h_img, w_img = img_uint8.shape[:2]
    margin_x = int(w * margin_pct)
    margin_y = int(h * margin_pct)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w_img, x + w + margin_x)
    y2 = min(h_img, y + h + margin_y)

    cropped = img_uint8[y1:y2, x1:x2]
    return cropped, (x1, y1, x2 - x1, y2 - y1)


def get_retina_binary_mask(
    image_rgb: np.ndarray,
    black_threshold: int = 15,
    kernel_size: int = 11,
) -> np.ndarray:
    """Generate a clean binary mask covering exclusively the retina region of interest.

    Args:
        image_rgb: Input RGB image (H, W, 3).
        black_threshold: Background intensity threshold.
        kernel_size: Morphological closing kernel size to fill interior gaps.

    Returns:
        Float32 binary mask array of shape (H, W) with values 1.0 (inside retina) and 0.0 (background).
    """
    if image_rgb.dtype != np.uint8:
        img_uint8 = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
    else:
        img_uint8 = image_rgb

    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY)

    # Morphological closing to fill interior blood vessels and dark lesions inside retina
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Keep only the largest connected component
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray, dtype=np.uint8)

    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_cnt], -1, 255, thickness=cv2.FILLED)

    mask_float = (mask / 255.0).astype(np.float32)
    return mask_float
