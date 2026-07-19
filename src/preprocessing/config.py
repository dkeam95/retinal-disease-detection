"""Configuration adapter for the preprocessing module.

This module provides convenient access to preprocessing-related
configuration parameters. """


from __future__ import annotations

from common.config.types import PreprocessingConfig


class PreprocessingSettings:
    """Convenient wrapper around PreprocessingConfig."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self._config = config


    @property
    def image_size(self) -> int:
        """Image size."""
        return self._config.image_size

    
    @property
    def mean(self) -> tuple[float, float, float]:
        """Mean values for each channel."""
        return self._config.mean

    
    @property
    def std(self) -> tuple[float, float, float]:
        """Standard deviation values for each channel."""
        return self._config.std
    

    @property
    def horizontal_flip_prob(self) -> float:
        """Probability of horizontal flip."""
        return self._config.horizontal_flip_prob

    
    @property
    def vertical_flip_prob(self) -> float:
        """Probability of vertical flip."""
        return self._config.vertical_flip_prob

    
    @property
    def rotation_limit(self) -> int:
        """Range of rotation in degrees."""
        return self._config.rotation_limit

    @property
    def rotation_prob(self) -> float:
        """Probability of rotation."""
        return self._config.rotation_prob


    @property
    def brightness_contrast_prob(self) -> float:
        """Probability of brightness contrast adjustment."""
        return self._config.brightness_contrast_prob

    