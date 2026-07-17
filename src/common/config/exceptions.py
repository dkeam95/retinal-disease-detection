"""Custom exceptions for the configuration system.

This module defines all exceptions related to loading,
parsing, and validating configuration files.
"""


class ConfigurationError(Exception):
    """Base exception for all configuration-related errors."""

    pass


class ConfigFileNotFoundError(ConfigurationError):
    """Raised when the configuration file does not exist."""

    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when the configuration file has an invalid
    structure or contains invalid values."""

    pass


class ConfigurationParsingError(ConfigurationError):
    """Raised when a configuration file cannot be parsed."""

    pass
