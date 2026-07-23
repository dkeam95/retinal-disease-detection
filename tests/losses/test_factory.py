import pytest
from torch import rand

from common.config.types import LossConfig
from losses.factory import build_loss
from losses.exceptions import UnknownLossError


def test_build_cross_entropy() -> None:
    config = LossConfig(
        name="cross_entropy",
        reduction="mean",
    )

    loss = build_loss(config)

    assert loss is not None


def test_unknown_loss() -> None:
    config = LossConfig(
        name="invalid_loss",
        reduction="mean",
    )

    with pytest.raises(
        UnknownLossError,
    ):
        build_loss(config)


def test_build_weighted_cross_entropy() -> None:
    config = LossConfig(
        name="weighted_cross_entropy",
        reduction="mean",
    )

    weights = rand(5)

    loss = build_loss(
        config,
        weights,
    )

    assert loss is not None