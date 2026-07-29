"""Custom exceptions for the configuration system.

This module defines all exceptions related to loading,
parsing, and validating configuration files.
"""

from pathlib import Path
from typing import Any


class ConfigurationError(Exception):
    """Base exception for all configuration-related errors."""

    pass


class ConfigFileNotFoundError(ConfigurationError, FileNotFoundError):
    """Raised when the specified configuration file does not exist on disk."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        message = f"Configuration file not found: '{self.file_path.absolute()}'"
        super().__init__(message)


class ConfigurationParsingError(ConfigurationError):
    """Raised when a configuration file cannot be parsed as valid YAML/JSON."""

    def __init__(self, file_path: str | Path, details: str) -> None:
        self.file_path = Path(file_path)
        self.details = details
        message = (
            f"Failed to parse configuration file '{self.file_path.name}': {details}"
        )
        super().__init__(message)


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration values or structure fail schema validation."""

    def __init__(self, param_name: str, value: Any, reason: str) -> None:
        self.param_name = param_name
        self.value = value
        self.reason = reason
        message = (
            f"Invalid configuration parameter '{param_name}' with value '{value}'. "
            f"Reason: {reason}"
        )
        super().__init__(message)
