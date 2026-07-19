"""
Type definitions for the preprocessing module.

This module contains shared type aliases used by the preprocessing
pipeline.
"""

from __future__ import annotations

from numpy.typing import NDArray
import numpy as np
from torch import Tensor

# Input image before preprocessing.
ImageArray = NDArray[np.uint8]

# Output tensor after preprocessing.
ImageTensor = Tensor