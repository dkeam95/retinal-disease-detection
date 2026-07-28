"""Unit tests for Diabetic Retinopathy class definitions."""

from src.common.classes import CLASS_NAMES, DRClass


def test_dr_class_enum_values() -> None:
    """Test DRClass Enum values and ordering."""
    assert DRClass.NO_DR == 0
    assert DRClass.MILD_NPDR == 1
    assert DRClass.MODERATE_NPDR == 2
    assert DRClass.SEVERE_NPDR == 3
    assert DRClass.PROLIFERATIVE_DR == 4
    assert len(DRClass) == 5


def test_class_names_alignment() -> None:
    """Test that CLASS_NAMES matches DRClass members in size and index mapping."""
    assert len(CLASS_NAMES) == len(DRClass)

    # Ensure index access directly maps to Enum values
    assert CLASS_NAMES[DRClass.NO_DR] == "No Diabetic Retinopathy"
    assert CLASS_NAMES[DRClass.MILD_NPDR] == "Mild Nonproliferative DR"
    assert CLASS_NAMES[DRClass.MODERATE_NPDR] == "Moderate Nonproliferative DR"
    assert CLASS_NAMES[DRClass.SEVERE_NPDR] == "Severe Nonproliferative DR"
    assert CLASS_NAMES[DRClass.PROLIFERATIVE_DR] == "Proliferative DR"


def test_dr_class_int_behavior() -> None:
    """Test that DRClass instances behave like standard integers (IntEnum)."""
    assert isinstance(DRClass.NO_DR, int)
    assert DRClass.NO_DR + 1 == DRClass.MILD_NPDR
    assert DRClass.PROLIFERATIVE_DR > DRClass.SEVERE_NPDR