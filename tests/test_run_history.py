"""Unit tests for the slim run-history persistence layer.

The master DB now holds only metadata + a pointer to a per-run
GeoPackage written by :mod:`omrat_utils.run_persistence` (which is
QGIS-only and tested separately).  The tests below exercise just the
metadata side, using stdlib ``sqlite3`` -- no QGIS needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.run_history import (  # noqa: E402
    RunHistory, RunMeta, default_db_path,
    totals_from_calc, slug, make_run_filename, _f,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_history(tmp_path: Path) -> RunHistory:
    return RunHistory(tmp_path / 'history.gpkg')


@pytest.fixture
def sample_totals() -> dict:
    return {
        'drift_allision':       1.5e-2,
        'drift_grounding':      2.0e-3,
        'drift_anchoring':      5.0e-4,
        'powered_grounding':    1.0e-4,
        'powered_allision':     8.0e-6,
        'head_on':              4.0e-7,
        'overtaking':           9.0e-7,
        'crossing':             5.0e-7,
        'bend':                 6.0e-7,
        'ship_collision_total': 2.4e-6,
    }


@pytest.fixture
def saved_run(tmp_history, sample_totals) -> tuple[RunHistory, int]:
    run_id = tmp_history.save_run(
        name='alpha',
        timestamp='2026-04-26 10:00:00',
        duration_seconds=12.34,
        totals=sample_totals,
        output_dir='/tmp/runs',
        output_filename='alpha_20260426_100000.gpkg',
    )
    return tmp_history, run_id


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

class TestDefaultDbPath:
    def test_returns_path_under_user_dir(self):
        path = default_db_path()
        # Master DB is plain SQLite -- no spatial features in here, so
        # the .sqlite extension correctly advertises that.
        assert path.name == 'omrat_history.sqlite'
        assert 'OMRAT' in path.parts
        plugin_dir = Path(__file__).resolve().parent.parent
        assert plugin_dir not in path.parents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestSlug:
    def test_strips_filesystem_unsafe_chars(self):
        assert slug('a/b\\c:d?e*f|g') == 'a_b_c_d_e_f_g'

    def test_collapses_whitespace(self):
        assert slug('hello  world\tagain') == 'hello_world_again'

    def test_empty_falls_back(self):
        assert slug('') == 'run'
        assert slug('   ') == 'run'


class TestMakeRunFilename:
    def test_format(self):
        import time
        # Build a known timestamp (UTC-naive struct_time).
        ts = time.strptime('2026-04-26 10:00:00', '%Y-%m-%d %H:%M:%S')
        assert make_run_filename('my model', ts) == 'my_model_20260426_100000.gpkg'

    def test_uses_now_when_ts_omitted(self):
        out = make_run_filename('x')
        assert out.startswith('x_')
        assert out.endswith('.gpkg')


class TestFloatCoercion:
    def test_none_passthrough(self):
        assert _f(None) is None

    def test_handles_non_numeric(self):
        assert _f('not a number') is None

    def test_floats_strings(self):
        assert _f('1.5') == 1.5
        assert _f(2) == 2.0


class TestTotalsFromCalc:
    def test_extracts_from_full_reports(self):
        class C:
            drifting_report = {'totals': {'allision': 1, 'grounding': 2, 'anchoring': 3}}
            collision_report = {'totals': {
                'head_on': 0.1, 'overtaking': 0.2,
                'crossing': 0.3, 'bend': 0.4, 'total': 1.0,
            }}
            powered_grounding_report = {'totals': {'grounding': 11}}
            powered_allision_report = {'totals': {'allision': 22}}
        out = totals_from_calc(C())
        assert out['drift_allision'] == 1
        assert out['drift_grounding'] == 2
        assert out['drift_anchoring'] == 3
        assert out['head_on'] == 0.1
        assert out['ship_collision_total'] == 1.0
        assert out['powered_grounding'] == 11
        assert out['powered_allision'] == 22

    def test_missing_reports_are_not_calculated(self):
        """A model the run skipped (Calculation matrix) has no report:
        its totals are ``None`` (empty history cell), never ``0``."""
        class C:
            pass
        out = totals_from_calc(C())
        assert out['drift_allision'] is None
        assert out['ship_collision_total'] is None
        assert out['powered_allision'] is None

    def test_partial_run_keeps_calculated_parts(self):
        class C:
            drifting_report = None
            collision_report = None
            powered_grounding_report = None
            powered_allision_report = {'totals': {'allision': 2.5e-4}}
        out = totals_from_calc(C())
        assert out['powered_allision'] == 2.5e-4
        assert out['drift_grounding'] is None
        assert out['head_on'] is None

    def test_empty_report_counts_as_zero(self):
        class C:
            drifting_report = {'totals': {}}
            collision_report = {}
            powered_grounding_report = None
            powered_allision_report = None
        out = totals_from_calc(C())
        assert out['drift_allision'] == 0.0
        assert out['bend'] == 0.0


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_creates_db_file(self, tmp_history):
        assert tmp_history.db_path.exists()

    def test_no_spatial_tables_in_master_db(self, tmp_history):
        """The master DB should hold only metadata; spatial features
        belong in per-run GeoPackages."""
        import sqlite3
        conn = sqlite3.connect(tmp_history.db_path)
        try:
            tables = {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        finally:
            conn.close()
        assert 'omrat_runs' in tables
        # No more spatial tables.
        for t in (
            'omrat_drifting_allision', 'omrat_drifting_grounding',
            'omrat_powered_grounding', 'omrat_powered_allision',
            'omrat_collision_lines', 'omrat_collision_points',
        ):
            assert t not in tables, f'{t} should have been removed'


# ---------------------------------------------------------------------------
# Save / list / get
# ---------------------------------------------------------------------------

class TestSaveAndList:
    def test_save_returns_run_id(self, saved_run):
        _, run_id = saved_run
        assert isinstance(run_id, int)
        assert run_id > 0

    def test_list_runs_empty_initially(self, tmp_history):
        assert tmp_history.list_runs() == []

    def test_list_runs_after_save(self, saved_run):
        history, run_id = saved_run
        runs = history.list_runs()
        assert len(runs) == 1
        assert isinstance(runs[0], RunMeta)
        assert runs[0].run_id == run_id
        assert runs[0].name == 'alpha'
        assert runs[0].duration_seconds == pytest.approx(12.34)
        assert runs[0].output_dir == '/tmp/runs'
        assert runs[0].output_filename == 'alpha_20260426_100000.gpkg'

    def test_list_runs_orders_newest_first(self, tmp_history):
        a = tmp_history.save_run(name='a')
        b = tmp_history.save_run(name='b')
        c = tmp_history.save_run(name='c')
        ids = [r.run_id for r in tmp_history.list_runs()]
        assert ids == [c, b, a]

    def test_save_persists_totals(self, saved_run):
        history, _ = saved_run
        run = history.list_runs()[0]
        assert run.drift_allision == pytest.approx(1.5e-2)
        assert run.head_on == pytest.approx(4.0e-7)
        assert run.bend == pytest.approx(6.0e-7)
        assert run.ship_collision_total == pytest.approx(2.4e-6)
        assert run.powered_grounding == pytest.approx(1.0e-4)
        assert run.powered_allision == pytest.approx(8.0e-6)

    def test_get_run_by_id(self, saved_run):
        history, run_id = saved_run
        run = history.get_run(run_id)
        assert run is not None
        assert run.name == 'alpha'

    def test_get_run_missing_returns_none(self, tmp_history):
        assert tmp_history.get_run(999) is None

    def test_save_with_minimal_args(self, tmp_history):
        run_id = tmp_history.save_run(name='x')
        run = tmp_history.get_run(run_id)
        assert run.name == 'x'
        assert run.duration_seconds is None
        assert run.drift_allision is None  # no totals dict provided


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

class TestCompare:
    def test_compare_runs_returns_in_input_order(self, tmp_history):
        a = tmp_history.save_run(name='a', totals={'drift_allision': 1.0})
        b = tmp_history.save_run(name='b', totals={'drift_allision': 2.0})
        out = tmp_history.compare_runs([b, a])
        assert [r.name for r in out] == ['b', 'a']
        assert out[0].drift_allision == pytest.approx(2.0)

    def test_compare_runs_skips_missing_ids(self, tmp_history):
        a = tmp_history.save_run(name='a')
        out = tmp_history.compare_runs([a, 9999])
        assert [r.run_id for r in out] == [a]


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_removes_metadata(self, saved_run):
        history, run_id = saved_run
        history.delete_run(run_id)
        assert history.list_runs() == []

    def test_delete_only_targeted_run(self, tmp_history):
        a = tmp_history.save_run(name='a')
        b = tmp_history.save_run(name='b')
        tmp_history.delete_run(a)
        remaining = tmp_history.list_runs()
        assert [r.run_id for r in remaining] == [b]

    def test_delete_removes_gpkg_when_requested(self, tmp_path):
        history = RunHistory(tmp_path / 'h.gpkg')
        gpkg = tmp_path / 'runs' / 'a.gpkg'
        gpkg.parent.mkdir()
        gpkg.write_bytes(b'fake gpkg')
        run_id = history.save_run(
            name='a',
            output_dir=str(gpkg.parent),
            output_filename=gpkg.name,
        )
        history.delete_run(run_id, delete_gpkg=True)
        assert not gpkg.exists()

    def test_delete_keeps_gpkg_by_default(self, tmp_path):
        history = RunHistory(tmp_path / 'h.gpkg')
        gpkg = tmp_path / 'runs' / 'a.gpkg'
        gpkg.parent.mkdir()
        gpkg.write_bytes(b'fake gpkg')
        run_id = history.save_run(
            name='a',
            output_dir=str(gpkg.parent),
            output_filename=gpkg.name,
        )
        history.delete_run(run_id)  # delete_gpkg=False (default)
        assert gpkg.exists()


# ---------------------------------------------------------------------------
# RunMeta.gpkg_path()
# ---------------------------------------------------------------------------

class TestGpkgPath:
    def test_returns_full_path(self):
        m = RunMeta(
            run_id=1, name='x', timestamp='t',
            output_dir='/tmp/runs', output_filename='a.gpkg',
        )
        assert m.gpkg_path() == Path('/tmp/runs/a.gpkg')

    def test_none_when_either_missing(self):
        assert RunMeta(run_id=1, name='x', timestamp='t').gpkg_path() is None
        assert RunMeta(
            run_id=1, name='x', timestamp='t', output_dir='/tmp/x',
        ).gpkg_path() is None
        assert RunMeta(
            run_id=1, name='x', timestamp='t', output_filename='a.gpkg',
        ).gpkg_path() is None


# ---------------------------------------------------------------------------
# Project tag + OMRAT version (side table ``omrat_run_info``)
# ---------------------------------------------------------------------------

# ``omrat_runs`` exactly as OMRAT <= 0.16.10 created it.
_OLD_SCHEMA = """
CREATE TABLE omrat_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
    timestamp TEXT NOT NULL, duration_seconds REAL, drift_allision REAL,
    drift_grounding REAL, drift_anchoring REAL, powered_grounding REAL,
    powered_allision REAL, head_on REAL, overtaking REAL, crossing REAL,
    merging REAL, bend REAL, ship_collision_total REAL, output_dir TEXT,
    output_filename TEXT, notes TEXT
)
"""
_OLD_COLUMNS = [
    'run_id', 'name', 'timestamp', 'duration_seconds', 'drift_allision',
    'drift_grounding', 'drift_anchoring', 'powered_grounding', 'powered_allision',
    'head_on', 'overtaking', 'crossing', 'merging', 'bend',
    'ship_collision_total', 'output_dir', 'output_filename', 'notes',
]


def _old_history(path: Path, output_dir: str) -> Path:
    """A history database written before the project / version columns."""
    import sqlite3
    conn = sqlite3.connect(path)
    try:
        conn.execute(_OLD_SCHEMA)
        conn.execute(
            "INSERT INTO omrat_runs (name, timestamp, drift_allision, output_dir, output_filename) "
            "VALUES ('old run', '2026-08-01 10:00:00', 1e-4, ?, 'old_run_20260801_100000.gpkg')",
            (output_dir,),
        )
        conn.commit()
    finally:
        conn.close()
    return path


class TestExistingHistories:
    """Other users' history files must keep working unchanged."""

    def test_old_database_opens_with_runs_intact(self, tmp_path):
        path = _old_history(tmp_path / 'old.sqlite', 'C:\\work\\Kattegat_2025')
        runs = RunHistory(path).list_runs()
        assert len(runs) == 1
        run = runs[0]
        assert run.name == 'old run'
        assert run.drift_allision == 1e-4
        assert run.project is None and run.omrat_version is None
        # Untagged: grouped under the name of its results folder.
        assert run.project_label() == 'Kattegat_2025'

    def test_runs_table_keeps_its_old_columns(self, tmp_history):
        """An older OMRAT reads ``SELECT * FROM omrat_runs`` into its
        RunMeta -- an extra column there would break it."""
        import sqlite3
        tmp_history.save_run('x', project='P', omrat_version='0.16.11')
        conn = sqlite3.connect(tmp_history.db_path)
        try:
            cols = [r[1] for r in conn.execute('PRAGMA table_info(omrat_runs)')]
        finally:
            conn.close()
        assert cols == _OLD_COLUMNS

    def test_old_database_accepts_new_runs_and_tags(self, tmp_path):
        path = _old_history(tmp_path / 'old.sqlite', '/data/runs/Baltic')
        history = RunHistory(path)
        new_id = history.save_run('new', output_dir='/data/runs/Baltic',
                                  project='Baltic study', omrat_version='0.16.11')
        old_id = next(r.run_id for r in history.list_runs() if r.name == 'old run')
        history.set_project([old_id], 'Baltic study')
        labels = {r.name: r.project_label() for r in history.list_runs()}
        assert labels == {'new': 'Baltic study', 'old run': 'Baltic study'}
        assert history.get_run(new_id).omrat_version == '0.16.11'


class TestProjectTag:
    def test_save_with_project_and_version(self, tmp_history):
        run_id = tmp_history.save_run('a', output_dir='/x/folder', project='  ESCOW  ',
                                      omrat_version='0.16.11')
        run = tmp_history.get_run(run_id)
        assert run.project == 'ESCOW'
        assert run.project_label() == 'ESCOW'
        assert run.omrat_version == '0.16.11'

    def test_empty_project_is_untagged(self, tmp_history):
        run_id = tmp_history.save_run('a', output_dir='/x/folder', project='   ')
        run = tmp_history.get_run(run_id)
        assert run.project is None
        assert run.project_label() == 'folder'

    def test_set_project_retags_and_clears(self, tmp_history):
        a = tmp_history.save_run('a', output_dir='/x/one', project='Wrong', omrat_version='0.16.9')
        b = tmp_history.save_run('b', output_dir='/x/two')
        tmp_history.set_project([a, b], 'Right')
        assert {r.project_label() for r in tmp_history.list_runs()} == {'Right'}
        # The version survives a re-tag.
        assert tmp_history.get_run(a).omrat_version == '0.16.9'
        tmp_history.set_project([a], '')
        assert tmp_history.get_run(a).project_label() == 'one'

    def test_compare_runs_carries_the_tag(self, tmp_history):
        a = tmp_history.save_run('a', project='P')
        assert tmp_history.compare_runs([a])[0].project == 'P'

    def test_delete_removes_the_tag(self, tmp_history):
        import sqlite3
        a = tmp_history.save_run('a', project='P', omrat_version='1')
        tmp_history.delete_run(a)
        conn = sqlite3.connect(tmp_history.db_path)
        try:
            n = conn.execute('SELECT COUNT(*) FROM omrat_run_info').fetchone()[0]
        finally:
            conn.close()
        assert n == 0

    def test_no_folder_no_tag(self):
        from omrat_utils.run_history import NO_PROJECT_LABEL
        assert RunMeta(run_id=1, name='x', timestamp='t').project_label() == NO_PROJECT_LABEL

    def test_project_counts_sorted_case_insensitively(self):
        from omrat_utils.run_history import project_counts
        runs = [
            RunMeta(run_id=1, name='a', timestamp='t', output_dir='/r/beta'),
            RunMeta(run_id=2, name='b', timestamp='t', project='Alpha'),
            RunMeta(run_id=3, name='c', timestamp='t', output_dir='D:\\r\\beta'),
        ]
        assert project_counts(runs) == {'Alpha': 1, 'beta': 2}


class TestFolderProject:
    def test_windows_and_posix_paths(self):
        from omrat_utils.run_history import folder_project
        assert folder_project('E:\\dev_e\\project\\ESCOW_case_study') == 'ESCOW_case_study'
        assert folder_project('/home/u/runs/Kattegat/') == 'Kattegat'
        assert folder_project('') == ''
        assert folder_project(None) == ''


class TestVersionBackfill:
    def _run_with_snapshot(self, history, out_dir: Path, text: str | None) -> int:
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = history.save_run('r', output_dir=str(out_dir), output_filename='r.gpkg')
        if text is not None:
            (out_dir / 'r.omrat').write_text(text, encoding='utf-8')
        return run_id

    def test_snapshot_version_reads_the_head(self, tmp_path):
        from omrat_utils.run_history import snapshot_version
        p = tmp_path / 's.omrat'
        p.write_text('{\n  "omrat_version": "0.16.7",\n  "pc": {}\n}', encoding='utf-8')
        assert snapshot_version(p) == '0.16.7'
        p.write_text('{"pc": {}}', encoding='utf-8')
        assert snapshot_version(p) == ''
        assert snapshot_version(tmp_path / 'missing.omrat') == ''

    def test_backfill_from_snapshots(self, tmp_history, tmp_path):
        with_v = self._run_with_snapshot(tmp_history, tmp_path / 'a', '{"omrat_version": "0.16.2"}')
        without = self._run_with_snapshot(tmp_history, tmp_path / 'b', '{"pc": {}}')
        missing = self._run_with_snapshot(tmp_history, tmp_path / 'c', None)
        runs = tmp_history.list_runs()
        assert tmp_history.backfill_versions(runs) == 2
        assert {r.run_id: r.omrat_version for r in runs}[with_v] == '0.16.2'
        stored = {r.run_id: r.omrat_version for r in tmp_history.list_runs()}
        assert stored == {with_v: '0.16.2', without: '', missing: None}

    def test_backfill_reads_each_snapshot_once(self, tmp_history, tmp_path, monkeypatch):
        import omrat_utils.run_history as rh
        self._run_with_snapshot(tmp_history, tmp_path / 'a', '{"pc": {}}')
        tmp_history.backfill_versions(tmp_history.list_runs())
        reads = []
        monkeypatch.setattr(rh, 'snapshot_version', lambda p: reads.append(p) or '')
        assert tmp_history.backfill_versions(tmp_history.list_runs()) == 0
        assert reads == []
