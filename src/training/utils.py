"""
Utility functions for the training module.

This module provides helper functions shared across training factories.
"""

from __future__ import annotations

from typing import Iterable


def normalize_name(name: str) -> str:
    """
    Normalize a component name.

    Parameters
    ----------
    name : str
        Component name.

    Returns
    -------
    str
        Normalized component name.
    """

    return name.strip().lower()


def validate_component_name(
    name: str,
    supported: Iterable[str],
) -> None:
    """
    Validate that a component name is supported.

    Parameters
    ----------
    name : str
        Requested component name.

    supported : Iterable[str]
        Collection of supported component names.

    Raises
    ------
    ValueError
        If the component is not supported.
    """

    normalized_name = normalize_name(name)

    normalized_supported = {
        normalize_name(component)
        for component in supported
    }

    if normalized_name not in normalized_supported:
        available = ", ".join(sorted(normalized_supported))

        raise ValueError(
            f"Unsupported component '{name}'. "
            f"Available: {available}"
        )


def is_supported_component(
    name: str,
    supported: Iterable[str],
) -> bool:
    """
    Check whether a component is supported.

    Parameters
    ----------
    name : str
        Component name.

    supported : Iterable[str]
        Supported component names.

    Returns
    -------
    bool
        True if the component is supported.
    """

    return (
        normalize_name(name)
        in {
            normalize_name(component)
            for component in supported
        }
    )