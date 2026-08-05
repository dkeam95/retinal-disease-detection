"""
Unit tests for checkpoint utility functions.

This module verifies filename generation, checkpoint validation,
directory creation, and latest checkpoint discovery.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from checkpoint.exceptions import (
    CheckpointNotFoundError,
    InvalidCheckpointError,
)
from checkpoint.utils import (
    build_checkpoint_filename,
    create_checkpoint_directory,
    find_latest_checkpoint,
    validate_checkpoint,
)


def test_build_checkpoint_filename() -> None:
    """
    Verify checkpoint filename generation.
    """

    filename = build_checkpoint_filename(
        epoch=12,
        metric=0.91234,
    )

    assert filename == "epoch_012_metric_0.9123.pt"


def test_create_checkpoint_directory(
    tmp_path: Path,
) -> None:
    """
    Verify checkpoint directory creation.
    """

    checkpoint_directory = (
        tmp_path / "checkpoints"
    )

    create_checkpoint_directory(
        checkpoint_directory,
    )

    assert checkpoint_directory.exists()
    assert checkpoint_directory.is_dir()


def test_validate_checkpoint_success(
    tmp_path: Path,
) -> None:
    """
    Verify validation succeeds for an existing checkpoint.
    """

    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()

    validate_checkpoint(checkpoint)


def test_validate_checkpoint_missing(
    tmp_path: Path,
) -> None:
    """
    Verify validation fails when checkpoint does not exist.
    """

    with pytest.raises(
        CheckpointNotFoundError,
    ):
        validate_checkpoint(
            tmp_path / "missing.pt",
        )


def test_validate_checkpoint_not_file(
    tmp_path: Path,
) -> None:
    """
    Verify validation fails for directories.
    """

    directory = tmp_path / "checkpoint"
    directory.mkdir()

    with pytest.raises(
        InvalidCheckpointError,
    ):
        validate_checkpoint(directory)


def test_find_latest_checkpoint(
    tmp_path: Path,
) -> None:
    """
    Verify latest checkpoint discovery.
    """

    (tmp_path / "epoch_001_metric_0.8000.pt").touch()
    latest = (
        tmp_path / "epoch_002_metric_0.9000.pt"
    )
    latest.touch()

    result = find_latest_checkpoint(
        tmp_path,
    )

    assert result == latest


def test_find_latest_checkpoint_empty(
    tmp_path: Path,
) -> None:
    """
    Verify None is returned for an empty directory.
    """

    result = find_latest_checkpoint(
        tmp_path,
    )

    assert result is None


def test_clean_state_dict() -> None:
    """
    Verify state dict key cleaning.
    """
    from checkpoint.utils import clean_state_dict

    state_dict = {
        "_orig_mod.backbone.weight": 1,
        "module.classifier.bias": 2,
        "normal_layer.weight": 3,
    }

    cleaned = clean_state_dict(state_dict)

    assert cleaned == {
        "backbone.weight": 1,
        "classifier.bias": 2,
        "normal_layer.weight": 3,
    }

