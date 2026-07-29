"""
Loss module exceptions.
"""

from __future__ import annotations


class LossError(Exception):
    """
    Base exception for all loss-related errors.
    """


class UnknownLossError(LossError):
    """
    Raised when an unknown loss function is requested.
    """


class LossInitializationError(LossError):
    """
    Raised when a loss function cannot be initialized.
    """
