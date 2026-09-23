"""Standalone tests for omrat_utils.number_input (no QGIS needed)."""
import pytest

from omrat_utils.number_input import (
    format_weight, normalise_weights, parse_decimal, weights_sum_ok,
)


@pytest.mark.parametrize("text, expected", [
    ("12.5", 12.5),
    ("12,5", 12.5),
    (" 7 ", 7.0),
    ("-0,25", -0.25),
    ("1e-3", 0.001),
    ("100", 100.0),
])
def test_parse_decimal_accepts_both_marks(text, expected):
    assert parse_decimal(text) == expected


def test_parse_decimal_empty_uses_default_or_raises():
    assert parse_decimal("", default=0.0) == 0.0
    assert parse_decimal("   ", default=3.0) == 3.0
    with pytest.raises(ValueError):
        parse_decimal("")


@pytest.mark.parametrize("text", ["abc", "1,234.5", "1.2.3", "12,,5"])
def test_parse_decimal_rejects_garbage(text):
    with pytest.raises(ValueError):
        parse_decimal(text)


def test_normalise_keeps_proportions_and_sums_to_target():
    out = normalise_weights([1, 2, 3, 4, 5, 6, 7, 8])
    assert sum(out) == pytest.approx(100.0, abs=1e-9)
    # proportions preserved: each is v/36*100
    for v, o in zip([1, 2, 3, 4, 5, 6, 7, 8], out):
        assert o == pytest.approx(v / 36 * 100, abs=0.011)


def test_normalise_rounded_values_sum_exactly():
    out = normalise_weights([1, 1, 1])
    assert out == [33.34, 33.33, 33.33] or sorted(out) == [33.33, 33.33, 33.34]
    assert round(sum(out), 2) == 100.0


def test_normalise_already_normalised_is_identity():
    vals = [12.5] * 8
    assert normalise_weights(vals) == vals
    assert weights_sum_ok(vals)
    assert not weights_sum_ok([12.5] * 7 + [12.0])


def test_normalise_all_zero_becomes_uniform():
    assert normalise_weights([0, 0, 0, 0]) == [25.0, 25.0, 25.0, 25.0]


def test_normalise_rejects_negative_and_handles_empty():
    with pytest.raises(ValueError):
        normalise_weights([1, -1])
    assert normalise_weights([]) == []


def test_normalise_other_target():
    assert normalise_weights([1, 3], target=1.0, ndigits=3) == [0.25, 0.75]


@pytest.mark.parametrize("value, text", [
    (12.5, "12.5"), (12.0, "12.0"), (0, "0.0"), (33.333, "33.33"), (100, "100.0"),
])
def test_format_weight(value, text):
    assert format_weight(value) == text
