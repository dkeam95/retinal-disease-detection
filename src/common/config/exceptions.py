"""Custom exceptions for the configuration system.

This module defines all exceptions related to loading,
parsing, and validating configuration files.
"""


class ConfigurationError(Exception):
    """Base exception for all configuration-related errors."""

    # Inherits from standard Python Exception to serve as a parent catch-all class
    pass


class ConfigFileNotFoundError(ConfigurationError):
    """Raised when the configuration file does not exist."""

    # Specific error triggered when the specified configuration file path is missing on disk
    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when the configuration file has an invalid
    structure or contains invalid values."""

    # Triggered when schema, missing keys, or parameter data types fail validation check
    pass


class ConfigurationParsingError(ConfigurationError):
    """Raised when a configuration file cannot be parsed."""

    # Triggered when the YAML/JSON parser encounters a syntax error while reading the file
    pass