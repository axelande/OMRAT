"""Unit tests for Storage._normalize_legacy_to_schema.

The normalisation path handles legacy .omrat files from multiple
historical OMRAT versions.  These tests pin the invariants so legacy
projects keep loading as the schema evolves.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.storage import Storage


@pytest.fixture
def storage():
    """Storage with a mocked parent -- we only exercise the pure
    normalisation method, so no QGIS state is needed."""
    return Storage(MagicMock())


# ---------------------------------------------------------------------------
# depths / objects normalisation: dict -> list form
# ---------------------------------------------------------------------------

class TestDepthsObjectsNormalisation:
    def test_dict_depth_becomes_list(self, storage):
        data = {'depths': [{'id': 'd1', 'depth': 6.0,
                            'polygon': 'POLYGON((0 0, 1 1, 1 0, 0 0))'}]}
        out = storage._normalize_legacy_to_schema(data)
        assert out['depths'] == [['d1', '6.0', 'POLYGON((0 0, 1 1, 1 0, 0 0))']]

    def test_list_depth_is_stringified(self, storage):
        data = {'depths': [['d1', 6, 'POLYGON(...)']]}
        out = storage._normalize_legacy_to_schema(data)
        # Values cast to str regardless of input type.
        assert out['depths'] == [['d1', '6', 'POLYGON(...)']]

    def test_dict_object_heights_fallback(self, storage):
        """Legacy schema used 'heights' (plural); the normaliser accepts it."""
        data = {'objects': [{'id': 'o1', 'heights': 12.0,
                             'polygon': 'POLYGON(...)'}]}
        out = storage._normalize_legacy_to_schema(data)
        assert out['objects'] == [['o1', '12.0', 'POLYGON(...)']]

    def test_malformed_entries_are_skipped(self, storage):
        data = {'depths': [['only', 'two']], 'objects': ['bogus']}
        out = storage._normalize_legacy_to_schema(data)
        assert out['depths'] == []
        assert out['objects'] == []


# ---------------------------------------------------------------------------
# segment_data: key mapping + defaults
# ---------------------------------------------------------------------------

class TestSegmentNormalisation:
    def test_space_keys_renamed_to_underscore(self, storage):
        data = {'segment_data': {'1': {'Start Point': 'a', 'End Point': 'b'}}}
        out = storage._normalize_legacy_to_schema(data)
        seg = out['segment_data']['1']
        assert seg['Start_Point'] == 'a' and 'Start Point' not in seg
        assert seg['End_Point'] == 'b' and 'End Point' not in seg

    def test_defaults_filled_in(self, storage):
        data = {'segment_data': {'7': {'Width': 1234}}}
        out = storage._normalize_legacy_to_schema(data)
        seg = out['segment_data']['7']
        assert seg['line_length'] == 0.0
        assert seg['Route_Id'] == 0
        assert seg['Leg_name'] == ''
        assert seg['Segment_Id'] == '7'
        assert seg['dist1'] == [] and seg['dist2'] == []
        assert seg['Width'] == 1234  # user value preserved

    def test_existing_segment_defaults_kept(self, storage):
        """`setdefault` must not overwrite an already-populated field."""
        data = {'segment_data': {'1': {'line_length': 999.0}}}
        out = storage._normalize_legacy_to_schema(data)
        assert out['segment_data']['1']['line_length'] == 999.0


# ---------------------------------------------------------------------------
# traffic_data: Ship Beam (meters) backfill
# ---------------------------------------------------------------------------

class TestTrafficDataNormalisation:
    def test_missing_beam_column_backfilled_with_zeros(self, storage):
        data = {
            'traffic_data': {
                '1': {
                    'East going': {
                        'Speed (knots)': [[10.0, 12.0], [5.0, 7.0]],
                    },
                },
            },
        }
        out = storage._normalize_legacy_to_schema(data)
        beam = out['traffic_data']['1']['East going']['Ship Beam (meters)']
        assert beam == [[0, 0], [0, 0]]

    def test_existing_beam_column_left_alone(self, storage):
        data = {
            'traffic_data': {
                '1': {'East going': {
                    'Speed (knots)': [[10.0]],
                    'Ship Beam (meters)': [[20.0]],
                }},
            },
        }
        out = storage._normalize_legacy_to_schema(data)
        assert (out['traffic_data']['1']['East going']['Ship Beam (meters)']
                == [[20.0]])


# ---------------------------------------------------------------------------
# drift.repair: use_lognormal/Normal migration + dist_type
# ---------------------------------------------------------------------------

class TestDriftRepairMigration:
    def test_normal_with_use_lognormal_true_is_converted(self, storage):
        """IWRAP-exported Normal repair with use_lognormal=True gets
        rewritten to an explicit Normal CDF func and use_lognormal=False.
        """
        data = {
            'drift': {
                'repair': {
                    'type': 'Normal', 'combi': '/Mean/Std. Dev.',
                    'use_lognormal': True,
                    'param_0': 0.2, 'param_1': 0.85,
                },
            },
        }
        out = storage._normalize_legacy_to_schema(data)
        repair = out['drift']['repair']
        assert repair['use_lognormal'] is False
        assert repair['dist_type'] == 'normal'
        assert repair['norm_mean'] == 0.2
        assert repair['norm_std'] == 0.85
        assert 'norm(loc=0.2, scale=0.85)' in repair['func']

    def test_non_normal_lognormal_kept(self, storage):
        data = {
            'drift': {
                'repair': {
                    'type': 'Lognormal', 'combi': '/Median/Shape',
                    'use_lognormal': True,
                    'std': 1.0, 'loc': 0.0, 'scale': 1.0,
                },
            },
        }
        out = storage._normalize_legacy_to_schema(data)
        # Untouched (still marked lognormal).
        assert out['drift']['repair']['use_lognormal'] is True

    def test_zero_std_guarded_to_tiny_positive(self, storage):
        data = {
            'drift': {
                'repair': {
                    'type': 'Normal', 'use_lognormal': True,
                    'param_0': 0.0, 'param_1': 0.0,
                },
            },
        }
        out = storage._normalize_legacy_to_schema(data)
        assert out['drift']['repair']['norm_std'] == 1e-6

    def test_use_lognormal_default_false_when_missing(self, storage):
        out = storage._normalize_legacy_to_schema({'drift': {}})
        assert out['drift']['repair']['use_lognormal'] is False


# ---------------------------------------------------------------------------
# drift: anchor_p / anchor_d / start_from / squat_mode defaults
# ---------------------------------------------------------------------------

class TestDriftDefaults:
    def test_anchor_p_default_when_missing(self, storage):
        out = storage._normalize_legacy_to_schema({'drift': {}})
        assert out['drift']['anchor_p'] == 0.7

    def test_anchor_p_percent_to_fraction(self, storage):
        """Legacy files sometimes stored anchor_p as a percentage."""
        out = storage._normalize_legacy_to_schema({'drift': {'anchor_p': 70}})
        assert out['drift']['anchor_p'] == 0.7

    def test_anchor_p_clamped_to_0_1(self, storage):
        # Negative gets clamped to 0.0.
        assert storage._normalize_legacy_to_schema(
            {'drift': {'anchor_p': -0.5}})['drift']['anchor_p'] == 0.0
        # Any value > 1.0 is assumed to be a stored percentage and divided
        # by 100, so 2.0 -> 0.02 (not clamped to 1.0).  150 -> 1.5 -> 0.015
        # after the /100 conversion; that's still in [0, 1] so no further clamp.
        assert storage._normalize_legacy_to_schema(
            {'drift': {'anchor_p': 2.0}})['drift']['anchor_p'] == pytest.approx(0.02)

    def test_anchor_p_percent_above_100_clamped_to_1(self, storage):
        # 200 treated as percentage -> 2.0, then clamped to 1.0 upper bound.
        assert storage._normalize_legacy_to_schema(
            {'drift': {'anchor_p': 200}})['drift']['anchor_p'] == 1.0

    def test_anchor_d_falls_back_to_anchor_depth(self, storage):
        out = storage._normalize_legacy_to_schema({'drift': {'anchor_depth': 5}})
        assert out['drift']['anchor_d'] == 5

    def test_start_from_default(self, storage):
        out = storage._normalize_legacy_to_schema({'drift': {}})
        assert out['drift']['start_from'] == 'leg_center'
        assert out['drift']['squat_mode'] == 'average_speed'


# ---------------------------------------------------------------------------
# drift.blackout_by_ship_type: defaults + JSON key coercion
# ---------------------------------------------------------------------------

class TestBlackoutByShipType:
    def test_defaults_used_when_missing(self, storage):
        out = storage._normalize_legacy_to_schema({'drift': {}})
        bo = out['drift']['blackout_by_ship_type']
        assert isinstance(bo, dict) and len(bo) > 0
        # IWRAP defaults: most 1.0, Passenger(17)/Ro-pax(18? etc.) lower.
        assert bo.get(18) == 1.0  # spot-check an int-keyed default

    def test_string_keys_converted_to_int(self, storage):
        data = {'drift': {'blackout_by_ship_type': {'0': '1.0', '17': '0.1'}}}
        out = storage._normalize_legacy_to_schema(data)
        bo = out['drift']['blackout_by_ship_type']
        assert bo == {0: 1.0, 17: 0.1}

    def test_non_coercible_entries_dropped(self, storage):
        data = {'drift': {'blackout_by_ship_type': {'0': 'not a number', '1': '0.5'}}}
        out = storage._normalize_legacy_to_schema(data)
        assert out['drift']['blackout_by_ship_type'] == {1: 0.5}

    def test_import_failure_falls_back_to_empty_dict(self, storage, monkeypatch):
        """If importing default_blackout_by_ship_type raises, drift
        ends up with an empty dict."""
        import builtins
        real_import = builtins.__import__

        def bad_import(name, *args, **kwargs):
            if name == 'compute.basic_equations':
                raise ImportError("synthetic")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', bad_import)
        out = storage._normalize_legacy_to_schema({'drift': {}})
        assert out['drift']['blackout_by_ship_type'] == {}


# ---------------------------------------------------------------------------
# Additional normalisation edge cases
# ---------------------------------------------------------------------------

class TestTrafficBeamFallback:
    def test_non_iterable_speed_row_yields_empty_beam_row(self, storage):
        """If a Speed (knots) row can't be iterated, the beam fallback adds []."""
        data = {
            'traffic_data': {
                '1': {
                    'East going': {
                        # A non-list row -> ``[0 for _ in row]`` raises TypeError.
                        'Speed (knots)': [None],
                    },
                },
            },
        }
        out = storage._normalize_legacy_to_schema(data)
        beam = out['traffic_data']['1']['East going']['Ship Beam (meters)']
        assert beam == [[]]


class TestAnchorPCoercionFailure:
    def test_non_numeric_anchor_p_falls_back_to_default(self, storage):
        """A non-float value for anchor_p triggers the except path -> 0.7."""
        out = storage._normalize_legacy_to_schema(
            {'drift': {'anchor_p': 'not-a-number'}})
        assert out['drift']['anchor_p'] == 0.7


class TestRepairMigrationExceptionPath:
    def test_repair_with_non_numeric_params_swallowed(self, storage):
        """If repair param casts raise, migration silently continues."""
        data = {
            'drift': {
                'repair': {
                    'type': 'Normal',
                    'use_lognormal': True,
                    'combi': 'Mean/Std',
                    'param_0': 'nope',  # float() raises
                    'param_1': 'also-nope',
                },
            },
        }
        out = storage._normalize_legacy_to_schema(data)
        # Migration skipped but the key still has use_lognormal default.
        assert out['drift']['repair']['use_lognormal'] is True


# ---------------------------------------------------------------------------
# store_all / select_file / load_from_path / load_all / new_file_path /
# last_used_dir -- exercise the public methods with mocked dialogs.
# ---------------------------------------------------------------------------

class TestPublicMethods:
    def _store_with_parent(self, tmp_path, monkeypatch, selected_path: str,
                           gather_data=None):
        """Helper: build a Storage whose GatherData.get_all_for_save returns
        a predictable dict, with the save dialog returning ``selected_path``."""
        from unittest.mock import MagicMock
        parent = MagicMock()
        parent.testing = False
        s = Storage(parent)

        fake_data = gather_data or {
            'pc': {}, 'drift': {'repair': {}}, 'segment_data': {},
            'traffic_data': {}, 'depths': [], 'objects': [],
            'ship_categories': {},
        }

        import omrat_utils.storage as storage_mod
        import omrat_utils.gather_data as gd_mod

        class FakeGather:
            def __init__(self, *a, **k):
                pass
            def get_all_for_save(self):
                return fake_data

        monkeypatch.setattr(storage_mod, 'GatherData', FakeGather)

        # Intercept new_file_path to return our chosen path tuple.
        monkeypatch.setattr(s, 'new_file_path',
                            lambda *a, **k: (selected_path, ''))
        monkeypatch.setattr(s, 'last_used_dir', lambda: str(tmp_path))
        return s, fake_data

    def test_store_all_writes_json(self, tmp_path, monkeypatch):
        out_path = str(tmp_path / 'proj.omrat')
        s, data = self._store_with_parent(tmp_path, monkeypatch, out_path)
        s.store_all()
        # File was written with the gathered JSON.
        assert (tmp_path / 'proj.omrat').exists()
        import json
        with open(out_path) as f:
            written = json.load(f)
        assert written == data

    def test_store_all_returns_silently_on_cancel(self, tmp_path, monkeypatch):
        """Empty file_path -> store_all returns without writing anything."""
        s, _ = self._store_with_parent(tmp_path, monkeypatch, '')
        # Should not raise / not write anything.
        s.store_all()
        assert list(tmp_path.iterdir()) == []

    def test_store_all_swallows_validation_error(self, tmp_path, monkeypatch, capsys):
        """If RootModelSchema raises, the error is printed and save continues."""
        out_path = str(tmp_path / 'bad.omrat')
        # Send truly invalid data so validation fails.
        bad = {'segment_data': 'not-a-dict'}
        s, _ = self._store_with_parent(
            tmp_path, monkeypatch, out_path, gather_data=bad)
        s.store_all()
        # File was still written (save path is after the validation try).
        assert (tmp_path / 'bad.omrat').exists()
        # Validation error got printed.
        err = capsys.readouterr().out
        assert err  # non-empty

    def test_select_file_testing_flag_returns_test_path(self):
        from unittest.mock import MagicMock
        parent = MagicMock()
        parent.testing = True
        path = Storage(parent).select_file()
        assert path.endswith('test_res.omrat')

    def test_select_file_production_uses_new_file_path(self, monkeypatch, tmp_path):
        from unittest.mock import MagicMock
        parent = MagicMock()
        parent.testing = False
        s = Storage(parent)
        monkeypatch.setattr(s, 'new_file_path',
                            lambda *a, **k: (str(tmp_path / 'x.omrat'), ''))
        monkeypatch.setattr(s, 'last_used_dir', lambda: str(tmp_path))
        assert s.select_file().endswith('x.omrat')

    def test_load_from_path_populates_via_gather(self, tmp_path, monkeypatch):
        """Valid .omrat file -> GatherData.populate is called with the
        normalised data."""
        import json
        file_path = tmp_path / 'good.omrat'
        file_path.write_text(json.dumps({'segment_data': {}, 'traffic_data': {}}))

        from unittest.mock import MagicMock
        parent = MagicMock()
        s = Storage(parent)

        populated = []

        class FakeGather:
            def __init__(self, *a, **k):
                pass
            def populate(self, data):
                populated.append(data)

        # Stub the schema validator so we exercise the success path without
        # building a fully-valid payload.
        import omrat_utils.storage as storage_mod
        monkeypatch.setattr(storage_mod, 'GatherData', FakeGather)
        fake_schema = MagicMock()
        fake_schema.model_validate.return_value = True
        monkeypatch.setattr(storage_mod, 'RootModelSchema', fake_schema)

        s.load_from_path(str(file_path))
        assert len(populated) == 1

    def test_load_from_path_validation_failure_returns_early(
        self, tmp_path, monkeypatch, capsys
    ):
        """Schema validation raises -> populate skipped, error printed."""
        import json
        file_path = tmp_path / 'bad.omrat'
        file_path.write_text(json.dumps({'segment_data': {}}))

        from unittest.mock import MagicMock
        parent = MagicMock()
        s = Storage(parent)

        populated = []

        class FakeGather:
            def __init__(self, *a, **k):
                pass
            def populate(self, data):
                populated.append(data)

        import omrat_utils.storage as storage_mod
        from pydantic import ValidationError

        monkeypatch.setattr(storage_mod, 'GatherData', FakeGather)

        fake_schema = MagicMock()
        fake_schema.model_validate.side_effect = ValidationError.from_exception_data(
            'root', [{'type': 'missing', 'loc': ('x',), 'input': None}],
        )
        monkeypatch.setattr(storage_mod, 'RootModelSchema', fake_schema)

        s.load_from_path(str(file_path))
        assert populated == []
        assert 'Validation error' in capsys.readouterr().out

    def test_load_all_no_file_returns_silently(self, monkeypatch):
        from unittest.mock import MagicMock
        parent = MagicMock()
        parent.testing = False
        s = Storage(parent)
        monkeypatch.setattr(s, 'select_file', lambda: '')
        # Should not raise.
        s.load_all()

    def test_load_all_delegates_to_load_from_path(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock
        parent = MagicMock()
        parent.testing = False
        s = Storage(parent)

        seen = []
        monkeypatch.setattr(s, 'select_file', lambda: str(tmp_path / 'f.omrat'))
        monkeypatch.setattr(s, 'load_from_path', lambda p: seen.append(p))
        s.load_all()
        assert seen == [str(tmp_path / 'f.omrat')]

    def test_new_file_path_save_branch(self, monkeypatch):
        from unittest.mock import MagicMock
        import omrat_utils.storage as storage_mod
        parent = MagicMock()
        s = Storage(parent)
        calls = []

        def fake_save(parent_widget, caption, default_path, filt):
            calls.append(('save', default_path))
            return ('/tmp/x.omrat', 'ok')

        monkeypatch.setattr(storage_mod.QFileDialog, 'getSaveFileName', fake_save)
        out = s.new_file_path(save=True, show_msg='Save', dir_path='/tmp',
                              generic_name='x.omrat', filter_text='*.omrat')
        assert out == ('/tmp/x.omrat', 'ok')
        assert calls[0][0] == 'save'

    def test_new_file_path_open_branch(self, monkeypatch):
        from unittest.mock import MagicMock
        import omrat_utils.storage as storage_mod
        parent = MagicMock()
        s = Storage(parent)
        called = []

        def fake_open(parent_widget, caption, default_path, filt):
            called.append(default_path)
            return ('/tmp/load.omrat', 'filter')

        monkeypatch.setattr(storage_mod.QFileDialog, 'getOpenFileName', fake_open)
        out = s.new_file_path(save=False, show_msg='Load', dir_path='/tmp',
                              generic_name='load.omrat', filter_text='*.omrat')
        assert out == ('/tmp/load.omrat', 'filter')
        assert called

    def test_new_file_path_empty_tuple_returns_empty_string(self, monkeypatch):
        """Falsy return from the dialog -> function returns ''."""
        from unittest.mock import MagicMock
        import omrat_utils.storage as storage_mod
        parent = MagicMock()
        s = Storage(parent)
        monkeypatch.setattr(
            storage_mod.QFileDialog, 'getSaveFileName',
            lambda *a, **k: (),  # empty tuple is falsy
        )
        out = s.new_file_path(save=True, show_msg='S', dir_path='/tmp',
                              generic_name='x.omrat', filter_text='*.omrat')
        assert out == ''

    def test_last_used_dir_reads_qsettings(self, monkeypatch):
        from unittest.mock import MagicMock
        import omrat_utils.storage as storage_mod
        parent = MagicMock()
        s = Storage(parent)

        fake_settings = MagicMock()
        fake_settings.value.return_value = '/my/last/dir'
        monkeypatch.setattr(storage_mod, 'QSettings', lambda *a, **k: fake_settings)
        assert s.last_used_dir() == '/my/last/dir'
        fake_settings.value.assert_called_once()
