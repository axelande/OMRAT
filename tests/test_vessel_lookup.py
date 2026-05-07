"""Headless tests for ``omrat_utils.vessel_lookup.VesselLookupConfig``.

Run with::

    /c/OSGeo4W/apps/Python312/python.exe -m pytest -p no:qgis --noconftest \\
        tests/test_vessel_lookup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.vessel_lookup import VesselLookupConfig


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestIsValid:
    def test_disabled_is_never_valid(self):
        # Even with all fields filled in, enabled=False short-circuits.
        cfg = VesselLookupConfig(
            enabled=False,
            schema="vessels", table="ship_registry", mmsi_col="mmsi",
            loa_col="loa",
        )
        assert not cfg.is_valid()

    def test_minimal_valid_config(self):
        cfg = VesselLookupConfig(
            enabled=True,
            schema="vessels", table="ship_registry", mmsi_col="mmsi",
            loa_col="loa",
        )
        assert cfg.is_valid()

    def test_blank_schema_invalid(self):
        cfg = VesselLookupConfig(
            enabled=True, schema="", table="t", mmsi_col="mmsi", loa_col="loa",
        )
        assert not cfg.is_valid()

    def test_blank_table_invalid(self):
        cfg = VesselLookupConfig(
            enabled=True, schema="s", table="", mmsi_col="mmsi", loa_col="loa",
        )
        assert not cfg.is_valid()

    def test_blank_mmsi_col_invalid(self):
        cfg = VesselLookupConfig(
            enabled=True, schema="s", table="t", mmsi_col="", loa_col="loa",
        )
        assert not cfg.is_valid()

    def test_all_output_columns_blank_invalid(self):
        """A JOIN that produces only NULLs for every output is pointless;
        treat it as 'not configured' so the caller skips the JOIN."""
        cfg = VesselLookupConfig(
            enabled=True, schema="s", table="t", mmsi_col="mmsi",
            loa_col="", beam_col="",
            ship_type_col="", air_draught_col="",
        )
        assert not cfg.is_valid()

    @pytest.mark.parametrize("bad", [
        "drop table foo;",       # statement injection
        "1col",                  # starts with digit
        "col-name",              # hyphen
        "schema.table",          # dotted (we accept only single-segment idents)
        '"quoted"',              # quotes
        "col;--",                # statement terminator
        "col with space",        # space
        "",                      # blank required-field
    ])
    def test_invalid_identifier_in_required_fields_rejected(self, bad):
        cfg = VesselLookupConfig(
            enabled=True, schema=bad, table="t", mmsi_col="mmsi", loa_col="loa",
        )
        assert not cfg.is_valid()

    @pytest.mark.parametrize("bad", [
        "drop table foo;", "1col", "col-name", "schema.table", '"x"',
    ])
    def test_invalid_identifier_in_optional_fields_rejected(self, bad):
        # Even an optional column is rejected when set to a non-identifier
        # string — better to refuse than to interpolate poison into SQL.
        cfg = VesselLookupConfig(
            enabled=True, schema="s", table="t", mmsi_col="mmsi",
            loa_col=bad,
        )
        assert not cfg.is_valid()

    def test_underscore_prefixed_identifiers_accepted(self):
        cfg = VesselLookupConfig(
            enabled=True, schema="_priv", table="_x", mmsi_col="_mmsi",
            loa_col="_loa",
        )
        assert cfg.is_valid()


# ---------------------------------------------------------------------------
# SQL builder
# ---------------------------------------------------------------------------


class TestBuildCte:
    def test_full_config_emits_all_columns(self):
        cfg = VesselLookupConfig(
            enabled=True, schema="vessels", table="ship_registry",
            mmsi_col="mmsi",
            loa_col="loa", beam_col="breadth_moulded",
            ship_type_col="ship_type", air_draught_col="height",
        )
        cte = cfg.build_cte()
        assert "external_vessels AS (" in cte
        assert "FROM vessels.ship_registry" in cte
        # Each requested source column is aliased to the stable ext_* name.
        assert "loa AS ext_loa" in cte
        assert "breadth_moulded AS ext_beam" in cte
        assert "ship_type AS ext_ship_type" in cte
        assert "height AS ext_air_draught" in cte

    def test_blank_optional_columns_become_typed_nulls(self):
        """A blank loa/beam/ship_type/air_draught column emits a typed
        NULL with the stable alias, so the caller's SELECT always
        resolves ``ext.ext_loa`` etc."""
        cfg = VesselLookupConfig(
            enabled=True, schema="s", table="t", mmsi_col="mmsi",
            loa_col="loa",  # only this one is set
            beam_col="", ship_type_col="", air_draught_col="",
        )
        cte = cfg.build_cte()
        assert "loa AS ext_loa" in cte
        assert "NULL::double precision AS ext_beam" in cte
        assert "NULL::int AS ext_ship_type" in cte
        assert "NULL::double precision AS ext_air_draught" in cte

    def test_mmsi_alias_uses_configured_column(self):
        """The MMSI column may have a project-specific name (e.g. 'imo'
        for an IMO-keyed table); the CTE re-aliases it to mmsi for the
        downstream JOIN."""
        cfg = VesselLookupConfig(
            enabled=True, schema="s", table="t", mmsi_col="imo_or_mmsi",
            loa_col="loa",
        )
        cte = cfg.build_cte()
        assert "imo_or_mmsi AS mmsi" in cte


# ---------------------------------------------------------------------------
# Round-trip helpers
# ---------------------------------------------------------------------------


class TestDictRoundTrip:
    def test_to_dict_then_from_dict_preserves_values(self):
        original = VesselLookupConfig(
            enabled=True, schema="s", table="t", mmsi_col="m",
            loa_col="l", beam_col="b",
            ship_type_col="st", air_draught_col="ad",
        )
        round_tripped = VesselLookupConfig.from_dict(original.to_dict())
        assert round_tripped == original

    def test_from_dict_filters_unknown_keys(self):
        cfg = VesselLookupConfig.from_dict({
            "enabled": True, "schema": "s", "table": "t",
            "mmsi_col": "m", "loa_col": "l",
            "stray": "should be ignored",
        })
        assert cfg.enabled is True
        assert cfg.schema == "s"
        # No error raised on the unknown 'stray' key.
