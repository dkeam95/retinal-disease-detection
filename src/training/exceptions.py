"""
Training module exceptions.

This module defines custom exceptions raised while constructing and
initializing the training pipeline.
"""

from __future__ import annotations


class TrainingError(Exception):
    """
    Base exception for all training module errors.
    """


class TrainingConfigurationError(TrainingError):
    """
    Raised when the training configuration is invalid.
    """


class FactoryError(TrainingError):
    """
    Base exception for all factory-related errors.
    """


class ModelFactoryError(FactoryError):
    """
    Raised when model creation fails.
    """


class OptimizerFactoryError(FactoryError):
    """
    Raised when optimizer creation fails.
    """


class SchedulerFactoryError(FactoryError):
    """
    Raised when scheduler creation fails.
    """


class LossFactoryError(FactoryError):
    """
    Raised when loss function creation fails.
    """