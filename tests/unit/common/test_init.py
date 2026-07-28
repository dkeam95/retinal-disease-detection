"""Unit tests for the top-level common module exports."""

import src.common as common


def test_common_exports() -> None:
    """Verify that all entities listed in __all__ are properly exported."""
    expected_exports = [
        "ConfigLoader",
        "validate_config",
        "ConfigurationError",
        "ConfigFileNotFoundError",
        "ConfigurationParsingError",
        "InvalidConfigurationError",
        "DatasetConfig",
        "ExperimentConfig",
        "ModelConfig",
        "PreprocessingConfig",
        "ProjectConfig",
        "TrainingConfig",
    ]

    assert hasattr(common, "__all__")
    assert sorted(common.__all__) == sorted(expected_exports)

    for export_name in common.__all__:
        assert hasattr(common, export_name), f"Missing export: {export_name}"