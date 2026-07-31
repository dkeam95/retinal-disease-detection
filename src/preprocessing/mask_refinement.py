"""Mask Edge Refinement and Guided Boundary Alignment Utilities.

Refines predicted lesion masks using image-guided contrast gradients and FOV constraints,
eliminating blurry boundary artifacts and ensuring sharp masks aligned with retinal pathology.
"""

from __future__ import annotations

import cv2
import numpy as np


class GuidedFilterMaskRefiner:
    """Post-processing refiner utilizing image intensity gradients to sharpen mask boundaries."""

    def __init__(
        self,
        radius: int = 4,
        eps: float = 1e-3,
        confidence_thresh: float = 0.50,
    ) -> None:
        """Initialize GuidedFilterMaskRefiner.

        Args:
            radius: Guided filter radius in pixels.
            eps: Regularization parameter for guided filter smoothing.
            confidence_thresh: Probability threshold for binary mask decision.
        """
        self.radius = radius
        self.eps = eps
        self.confidence_thresh = confidence_thresh

    def refine(
        self,
        image_rgb: np.ndarray,
        raw_mask: np.ndarray,
        fov_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sharpen mask boundary using guided filtering aligned with image color gradients.

        Args:
            image_rgb: Guidance RGB image (H, W, 3) uint8 or float32.
            raw_mask: Predicted probability mask (H, W) in range [0.0, 1.0] or uint8 [0, 255].
            fov_mask: Optional binary FOV mask (H, W) uint8 [0, 255].

        Returns:
            Refined binary mask (H, W) uint8 array [0, 255].
        """
        if image_rgb.dtype != np.uint8:
            guidance_gray = cv2.cvtColor(
                (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
            )
        else:
            guidance_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Normalize guidance and raw mask to float32 [0.0, 1.0]
        guidance_f32 = (guidance_gray / 255.0).astype(np.float32)

        if raw_mask.dtype == np.uint8:
            mask_f32 = (raw_mask / 255.0).astype(np.float32)
        else:
            mask_f32 = raw_mask.astype(np.float32)

        # Guided filter implementation using OpenCV ximgproc or standard bilateral fallback
        try:
            # OpenCV ximgproc guidedFilter if available
            guided = cv2.ximgproc.guidedFilter(
                guide=guidance_gray,
                src=(mask_f32 * 255.0).astype(np.uint8),
                radius=self.radius,
                eps=self.eps,
            )
            refined_prob = guided / 255.0
        except AttributeError:
            # Fallback: Bilateral filter on mask guided by contrast
            mask_u8 = (mask_f32 * 255.0).astype(np.uint8)
            bilateral = cv2.bilateralFilter(
                mask_u8, d=self.radius * 2 + 1, sigmaColor=75, sigmaSpace=75
            )
            refined_prob = bilateral / 255.0

        # Apply confidence threshold
        refined_binary = (refined_prob >= self.confidence_thresh).astype(np.uint8) * 255

        # Perform light morphological opening to eliminate isolated 1-pixel noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined_binary = cv2.morphologyEx(refined_binary, cv2.MORPH_OPEN, kernel)

        # Clip strictly inside FOV mask if provided
        if fov_mask is not None:
            fov_binary = (fov_mask > 0).astype(np.uint8) * 255
            refined_binary = cv2.bitwise_and(refined_binary, fov_binary)

        return refined_binary


def refine_lesion_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    confidence_thresh: float = 0.50,
    fov_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Utility function to sharpen lesion mask boundaries.

    Args:
        image_rgb: Input RGB image (H, W, 3).
        mask: Probability mask or binary mask (H, W).
        confidence_thresh: Probability threshold for binarization.
        fov_mask: Optional binary FOV mask.

    Returns:
        Refined binary mask (H, W) uint8 array [0, 255].
    """
    refiner = GuidedFilterMaskRefiner(confidence_thresh=confidence_thresh)
    return refiner.refine(image_rgb=image_rgb, raw_mask=mask, fov_mask=fov_mask)
