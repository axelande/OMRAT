# -*- coding: utf-8 -*-
"""IWRAP export must carry the project's own causation factors.

Before v0.14.0, ``_add_global_settings`` wrote a fixed block of constants
and never looked at ``data['pc']``.  An exported model was therefore
scored with IWRAP defaults instead of the values set under
**Settings -> Causation Factors** -- which silently invalidates any
OMRAT-vs-IWRAP comparison built on that export.
"""
import xml.etree.ElementTree as ET

import pytest

from compute.iwrap_convertion import (
    _CF_EXPORT_MAP,
    _CF_IMPORT_MAP,
    _add_global_settings,
    _parse_global_settings_el,
)


def _custom_pc() -> dict:
    """A ``pc`` block where no value equals the hardcoded default."""
    return {
        'headon': 7.7e-5,
        'overtaking': 2.2e-4,
        'crossing': 3.3e-4,
        'merging': 4.4e-5,
        'bend': 5.5e-4,
        'grounding': 6.6e-4,
        'allision': 8.8e-4,
        'grounding_drifting_rf': 0.5,
        'allision_drifting_rf': 0.25,
        'mean_time_between_checks': 180.0,
    }


def _export(data: dict) -> ET.Element:
    root = ET.Element('riskmodel')
    _add_global_settings(root, data)
    return root


def _causation_attrs(root: ET.Element) -> dict:
    cf = root.find('global_settings/causation_factors')
    assert cf is not None
    return dict(cf.attrib)


class TestExportUsesProjectValues:
    @pytest.mark.parametrize('iwrap_attr,pc_key', [
        ('p_headon_causation', 'headon'),
        ('p_overtaking_causation', 'overtaking'),
        ('p_crossing_causation', 'crossing'),
        ('p_merging_causation', 'merging'),
        ('p_bend_causation', 'bend'),
        ('p_grounding_causation', 'grounding'),
        ('p_grounding_no_turn_causation', 'grounding'),
        ('p_allision_causation', 'allision'),
        ('p_allision_no_turn_causation', 'allision'),
        ('p_grounding_drifting_causation', 'grounding_drifting_rf'),
        ('p_allision_drifting_causation', 'allision_drifting_rf'),
    ])
    def test_each_factor_is_exported(self, iwrap_attr, pc_key):
        pc = _custom_pc()
        attrs = _causation_attrs(_export({'pc': pc}))
        assert float(attrs[iwrap_attr]) == pytest.approx(pc[pc_key])

    def test_merging_is_not_the_old_hardcoded_constant(self):
        """The specific regression: merging was always '0.00013'."""
        attrs = _causation_attrs(_export({'pc': _custom_pc()}))
        assert attrs['p_merging_causation'] != '0.00013'
        assert float(attrs['p_merging_causation']) == pytest.approx(4.4e-5)

    def test_mean_time_between_checks_is_exported(self):
        root = _export({'pc': _custom_pc()})
        misc = root.find('global_settings/misc')
        assert misc is not None
        assert float(misc.get('meantime_between_checks')) == pytest.approx(180.0)


class TestExportFallbacks:
    def test_missing_pc_block_still_produces_valid_xml(self):
        attrs = _causation_attrs(_export({}))
        assert set(attrs) == set(_CF_EXPORT_MAP)
        for value in attrs.values():
            float(value)   # every attribute must parse as a number

    def test_absent_key_falls_back_to_the_documented_default(self):
        attrs = _causation_attrs(_export({'pc': {'headon': 1.0}}))
        assert float(attrs['p_headon_causation']) == pytest.approx(1.0)
        assert attrs['p_crossing_causation'] == _CF_EXPORT_MAP[
            'p_crossing_causation'
        ][1]

    def test_grounding_falls_back_to_p_pc(self):
        """An IWRAP import writes ``p_pc``, the dialog writes ``grounding``;
        export accepts either."""
        attrs = _causation_attrs(_export({'pc': {'p_pc': 9.9e-4}}))
        assert float(attrs['p_grounding_causation']) == pytest.approx(9.9e-4)

    def test_unparseable_value_falls_back_rather_than_raising(self):
        attrs = _causation_attrs(_export({'pc': {'crossing': 'not a number'}}))
        assert attrs['p_crossing_causation'] == _CF_EXPORT_MAP[
            'p_crossing_causation'
        ][1]

    def test_non_dict_pc_is_tolerated(self):
        attrs = _causation_attrs(_export({'pc': ['unexpected']}))
        assert set(attrs) == set(_CF_EXPORT_MAP)


class TestRoundTrip:
    """Export then re-import must return the same numbers.

    This is what keeps ``_CF_EXPORT_MAP`` and the importer's
    ``cf_mapping`` naming the same IWRAP attributes.
    """

    def test_collision_factors_survive_a_round_trip(self):
        pc = _custom_pc()
        root = _export({'pc': pc})
        result = {'pc': {}, 'segment_data': {}}
        _parse_global_settings_el(
            root.find('global_settings'), result, debug=False,
        )
        for key in ('headon', 'overtaking', 'crossing', 'merging', 'bend'):
            assert result['pc'][key] == pytest.approx(pc[key]), key

    def test_drifting_reduction_factors_survive_a_round_trip(self):
        pc = _custom_pc()
        root = _export({'pc': pc})
        result = {'pc': {}, 'segment_data': {}}
        _parse_global_settings_el(
            root.find('global_settings'), result, debug=False,
        )
        assert result['pc']['grounding_drifting_rf'] == pytest.approx(0.5)
        assert result['pc']['allision_drifting_rf'] == pytest.approx(0.25)

    def test_every_exported_attribute_is_understood_by_the_importer(self):
        """No exported attribute may be silently dropped on import."""
        root = _export({'pc': _custom_pc()})
        result = {'pc': {}, 'segment_data': {}}
        _parse_global_settings_el(
            root.find('global_settings'), result, debug=False,
        )
        # Every attribute we write must land somewhere in the pc dict.
        assert len(result['pc']) >= len(_CF_EXPORT_MAP)

    def test_allision_lands_on_the_key_the_model_reads(self):
        """``pc['allision']``, not ``pc['allision_pc']``.

        ``compute/powered_model.py`` reads ``pc_vals.get('allision', ...)``.
        The importer used to write ``allision_pc``, which nothing reads, so
        an imported IWRAP model quietly used OMRAT's default instead of the
        factor in the XML.
        """
        pc = _custom_pc()
        root = _export({'pc': pc})
        result = {'pc': {}, 'segment_data': {}}
        _parse_global_settings_el(
            root.find('global_settings'), result, debug=False,
        )
        assert result['pc']['allision'] == pytest.approx(pc['allision'])

    def test_grounding_round_trips_via_p_pc(self):
        """``powered_model`` falls back ``grounding`` -> ``p_pc``, so
        landing on ``p_pc`` is correct for grounding."""
        pc = _custom_pc()
        root = _export({'pc': pc})
        result = {'pc': {}, 'segment_data': {}}
        _parse_global_settings_el(
            root.find('global_settings'), result, debug=False,
        )
        effective = result['pc'].get('grounding', result['pc'].get('p_pc'))
        assert effective == pytest.approx(pc['grounding'])


class TestPoweredCategoryMapping:
    """IWRAP splits powered grounding / allision into two categories.

    ``p_*_causation`` is Category I (obstacle already in the lane) and
    ``p_*_no_turn_causation`` is Category II (a turn was required and the
    ship failed to make it).  OMRAT models only Category II, so its
    ``grounding`` / ``allision`` factors must reach the ``_no_turn``
    attributes -- before v0.14.0 they only reached the Category-I ones.
    """

    def test_no_turn_attributes_carry_the_project_factors(self):
        pc = _custom_pc()
        attrs = _causation_attrs(_export({'pc': pc}))
        assert float(attrs['p_grounding_no_turn_causation']) == pytest.approx(
            pc['grounding'])
        assert float(attrs['p_allision_no_turn_causation']) == pytest.approx(
            pc['allision'])

    def test_no_turn_attributes_are_not_the_old_constant(self):
        attrs = _causation_attrs(_export({'pc': _custom_pc()}))
        assert attrs['p_grounding_no_turn_causation'] != '0.000155'
        assert attrs['p_allision_no_turn_causation'] != '0.000155'

    def test_both_categories_get_the_same_value(self):
        """OMRAT has no separate Category-I input, so exporting a stale
        constant there would score geometry IWRAP computes anyway with a
        factor the user never chose."""
        attrs = _causation_attrs(_export({'pc': _custom_pc()}))
        assert (attrs['p_grounding_causation']
                == attrs['p_grounding_no_turn_causation'])
        assert (attrs['p_allision_causation']
                == attrs['p_allision_no_turn_causation'])

    def test_no_turn_wins_when_an_iwrap_file_sets_both(self):
        """Reading a hand-tuned IWRAP file: the turn-failure factor is the
        one OMRAT's powered models apply, so it must take precedence."""
        root = ET.Element('riskmodel')
        gs = ET.SubElement(root, 'global_settings')
        cf = ET.SubElement(gs, 'causation_factors')
        cf.set('p_grounding_causation', '1e-4')            # Category I
        cf.set('p_grounding_no_turn_causation', '9e-4')    # Category II
        cf.set('p_allision_causation', '2e-4')
        cf.set('p_allision_no_turn_causation', '8e-4')
        result = {'pc': {}, 'segment_data': {}}
        _parse_global_settings_el(gs, result, debug=False)
        assert result['pc']['grounding'] == pytest.approx(9e-4)
        assert result['pc']['allision'] == pytest.approx(8e-4)

    def test_category_one_alone_is_still_read(self):
        """An IWRAP file that only sets the Category-I attribute should
        still give OMRAT a value rather than falling back to a default."""
        root = ET.Element('riskmodel')
        gs = ET.SubElement(root, 'global_settings')
        cf = ET.SubElement(gs, 'causation_factors')
        cf.set('p_grounding_causation', '3.3e-4')
        result = {'pc': {}, 'segment_data': {}}
        _parse_global_settings_el(gs, result, debug=False)
        assert result['pc']['grounding'] == pytest.approx(3.3e-4)

    def test_p_pc_is_kept_in_step_with_grounding(self):
        """``p_pc`` is the dialog's Powered field and a fallback the
        grounding model reads; it must not show a stale default next to an
        imported grounding factor."""
        pc = _custom_pc()
        root = _export({'pc': pc})
        result = {'pc': {}, 'segment_data': {}}
        _parse_global_settings_el(
            root.find('global_settings'), result, debug=False,
        )
        assert result['pc']['p_pc'] == pytest.approx(pc['grounding'])


class TestMapsStayInStep:
    def test_every_import_attribute_is_exported(self):
        """A round-trip is only lossless if both maps name the same
        IWRAP attributes."""
        imported = {a for attrs in _CF_IMPORT_MAP.values() for a in attrs}
        assert imported <= set(_CF_EXPORT_MAP), (
            imported - set(_CF_EXPORT_MAP)
        )

    def test_every_exported_attribute_is_imported(self):
        exported = set(_CF_EXPORT_MAP)
        imported = {a for attrs in _CF_IMPORT_MAP.values() for a in attrs}
        assert exported <= imported, exported - imported
