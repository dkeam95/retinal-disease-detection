"""Experiments management and comparative analysis package."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .comparator import ExperimentComparator
    from .runner import ExperimentRunner

__all__ = ["ExperimentRunner", "ExperimentComparator"]


def __getattr__(name: str):
    if name == "ExperimentRunner":
        from .runner import ExperimentRunner

        return ExperimentRunner
    if name == "ExperimentComparator":
        from .comparator import ExperimentComparator

        return ExperimentComparator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
