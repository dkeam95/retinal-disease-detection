"""Configuration adapter for the preprocessing module.

This module provides convenient access to preprocessing-related
configuration parameters.
"""

from __future__ import annotations                   # Enables modern type hints (Python 3.7+)

from common.config.types import PreprocessingConfig  # Raw dataclass holding preprocessing parameters


class PreprocessingSettings:
    """Convenient wrapper around PreprocessingConfig."""

    def __init__(self, config: PreprocessingConfig) -> None:
        # Wrap the raw PreprocessingConfig object to provide controlled property access
        self._config = config

    @property
    def image_size(self) -> int:
        """Image size."""
        # Target spatial dimension (height and width) for resizing
        return self._config.image_size

    @property
    def mean(self) -> tuple[float, float, float]:
        """Mean values for each channel."""
        # Channel-wise mean values for RGB image normalization
        return self._config.mean

    @property
    def std(self) -> tuple[float, float, float]:
        """Standard deviation values for each channel."""
        # Channel-wise standard deviation values for RGB image normalization
        return self._config.std

    @property
    def horizontal_flip_prob(self) -> float:
        """Probability of horizontal flip."""
        # Augmentation probability for random horizontal flipping
        return self._config.horizontal_flip_prob

    @property
    def vertical_flip_prob(self) -> float:
        """Probability of vertical flip."""
        # Augmentation probability for random vertical flipping
        return self._config.vertical_flip_prob

    @property
    def rotation_limit(self) -> int:
        """Range of rotation in degrees."""
        # Maximum degrees for random image rotation (-limit, +limit)
        return self._config.rotation_limit

    @property
    def rotation_prob(self) -> float:
        """Probability of rotation."""
        # Augmentation probability for applying random rotation
        return self._config.rotation_prob

    @property
    def brightness_contrast_prob(self) -> float:
        """Probability of brightness contrast adjustment."""
        # Augmentation probability for applying random brightness/contrast adjustments
        return self._config.brightness_contrast_prob