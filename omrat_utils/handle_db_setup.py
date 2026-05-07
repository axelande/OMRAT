"""Multi-page database-setup wizard for OMRAT.

Drives the user through:

    Intro → Connection → Capabilities → Ingest → Done

The wizard owns a single :class:`ConnectionProfile` (plus an
:class:`IngestionSettings`) shared across pages.  Each page reads/writes
that state via ``self.wizard().profile`` / ``self.wizard().ingest_settings``.

Probe and migration operations run synchronously with a wait cursor — both
are sub-second for a healthy local DB.  Ingestion runs on a ``QThread``
(see :func:`omrat_utils.handle_ais_ingest.make_worker`) since it can take
minutes-to-hours on real datasets.

Wired into the QGIS menu by ``omrat.OMRAT.open_db_setup_wizard``.
"""
from __future__ import annotations

import os
from typing import Optional

from qgis.PyQt.QtCore import Qt, QUrl
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from omrat_utils.db_setup import (
    ConnectionProfile,
    DbProbe,
    IngestionSettings,
    MigrationError,
    Migrator,
    ProbeResult,
)


# Qt 6 namespaced enums; fall back to Qt 5 flat ones.
_WAIT_CURSOR = getattr(Qt, "WaitCursor", None) or Qt.CursorShape.WaitCursor
_PASSWORD_ECHO = getattr(QLineEdit, "Password", None) or QLineEdit.EchoMode.Password
_PASSWORD_ECHO_ON_EDIT = (
    getattr(QLineEdit, "PasswordEchoOnEdit", None) or QLineEdit.EchoMode.PasswordEchoOnEdit
)


def _ok(label: str) -> str:
    return f"✓ {label}"


def _bad(label: str) -> str:
    return f"✗ {label}"


def _info(label: str) -> str:
    return f"… {label}"


# ---------------------------------------------------------------------------
# Wizard
# ---------------------------------------------------------------------------


class DbSetupWizard(QWizard):
    """Top-level wizard.  Holds the shared connection + ingestion state."""

    PAGE_INTRO = 0
    PAGE_CONNECTION = 1
    PAGE_CAPABILITIES = 2
    PAGE_INGEST = 3
    PAGE_DONE = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OMRAT - Database setup")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle if hasattr(QWizard, "WizardStyle") else QWizard.ModernStyle)
        self.setMinimumSize(760, 760)

        # Shared state across pages.
        self.profile: ConnectionProfile = ConnectionProfile.from_qsettings()
        self.ingest_settings: IngestionSettings = IngestionSettings.from_qsettings(
            self.profile.name
        )
        self.last_probe: Optional[ProbeResult] = None

        self.setPage(self.PAGE_INTRO, IntroPage(self))
        self.setPage(self.PAGE_CONNECTION, ConnectionPage(self))
        self.setPage(self.PAGE_CAPABILITIES, CapabilityPage(self))
        self.setPage(self.PAGE_INGEST, IngestPage(self))
        self.setPage(self.PAGE_DONE, DonePage(self))


# ---------------------------------------------------------------------------
# Page 1 - Intro
# ---------------------------------------------------------------------------


class IntroPage(QWizardPage):
    """Brief overview + link to the Docker quickstart for users without a server."""

    def __init__(self, wizard: DbSetupWizard):
        super().__init__(wizard)
        self.setTitle("Set up an OMRAT database")
        self.setSubTitle(
            "OMRAT stores AIS-derived linestring segments in a PostgreSQL/PostGIS "
            "database.  This wizard checks an existing server, or helps you stand "
            "up a fresh one locally."
        )

        layout = QVBoxLayout(self)
        intro = QLabel(
            "<p><b>What this wizard does:</b></p>"
            "<ol>"
            "<li>Tests the connection to your Postgres server.</li>"
            "<li>Detects what's missing (PostGIS, OMRAT schema, year tables).</li>"
            "<li>Applies the missing pieces in idempotent steps.</li>"
            "</ol>"
            "<p>If you do not yet have a Postgres server, the easiest option is "
            "the bundled Docker stack.  Click below for the one-page quickstart, "
            "then return here to connect to <code>localhost</code>.</p>"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        btn_row = QHBoxLayout()
        btn_docker = QPushButton("Open Docker quickstart (README)")
        btn_docker.clicked.connect(self._open_docker_readme)
        btn_row.addWidget(btn_docker)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        layout.addStretch(1)

    def _open_docker_readme(self) -> None:
        # README sits at <repo>/docker/README.md.  Resolve relative to this
        # file so it works whether OMRAT is run from source or installed.
        readme = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "docker", "README.md")
        )
        if os.path.isfile(readme):
            QDesktopServices.openUrl(QUrl.fromLocalFile(readme))
        else:
            QMessageBox.information(
                self,
                "Docker quickstart",
                "The bundled docker/ directory was not found in this install.\n\n"
                "Visit the OMRAT repository for setup instructions:\n"
                "https://github.com/axelande/OMRAT",
            )


# ---------------------------------------------------------------------------
# Page 2 - Connection
# ---------------------------------------------------------------------------


class ConnectionPage(QWizardPage):
    """Connection form + Test button.  ``Next`` enables once a probe lands."""

    def __init__(self, wizard: DbSetupWizard):
        super().__init__(wizard)
        self.setTitle("Connect to the database")
        self.setSubTitle(
            "Enter the connection details.  Click Test to verify reachability "
            "and login.  The probe runs read-only and does not modify the database."
        )
        self._probe_ok = False

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.le_host = QLineEdit()
        self.sb_port = QSpinBox()
        self.sb_port.setRange(1, 65535)
        self.sb_port.setValue(5432)
        self.le_db = QLineEdit()
        self.le_user = QLineEdit()
        self.le_pass = QLineEdit()
        self.le_pass.setEchoMode(_PASSWORD_ECHO_ON_EDIT)
        self.le_schema = QLineEdit()
        self.le_schema.setText("omrat")
        self.le_sslmode = QLineEdit()
        self.le_sslmode.setText("prefer")

        form.addRow("Host", self.le_host)
        form.addRow("Port", self.sb_port)
        form.addRow("Database", self.le_db)
        form.addRow("User", self.le_user)
        form.addRow("Password", self.le_pass)
        form.addRow("Schema", self.le_schema)
        form.addRow("SSL mode", self.le_sslmode)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.btn_test = QPushButton("Test connection")
        self.btn_test.clicked.connect(self._on_test)
        btn_row.addWidget(self.btn_test)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        layout.addStretch(1)

    # -- QWizardPage hooks ---------------------------------------------------

    def initializePage(self) -> None:
        wiz = self._wiz()
        p = wiz.profile
        self.le_host.setText(p.host)
        self.sb_port.setValue(int(p.port) if p.port else 5432)
        self.le_db.setText(p.database)
        self.le_user.setText(p.user)
        self.le_pass.setText(p.password)
        self.le_schema.setText(p.schema or "omrat")
        self.le_sslmode.setText(p.sslmode or "prefer")
        self._probe_ok = bool(wiz.last_probe and wiz.last_probe.can_login)
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self._probe_ok

    def validatePage(self) -> bool:
        # Persist the form into the shared profile so later pages and saved
        # state see the latest values.  We do not write to QSettings yet —
        # that happens on the Done page after the user confirms.
        self._capture_into_profile()
        return True

    # -- helpers -------------------------------------------------------------

    def _wiz(self) -> DbSetupWizard:
        return self.wizard()  # type: ignore[return-value]

    def _capture_into_profile(self) -> None:
        p = self._wiz().profile
        p.host = self.le_host.text().strip()
        p.port = int(self.sb_port.value())
        p.database = self.le_db.text().strip()
        p.user = self.le_user.text().strip()
        p.password = self.le_pass.text()
        p.schema = self.le_schema.text().strip() or "omrat"
        p.sslmode = self.le_sslmode.text().strip() or "prefer"

    def _on_test(self) -> None:
        self._capture_into_profile()
        wiz = self._wiz()
        if not wiz.profile.is_complete():
            self.lbl_status.setText(_bad("Host, database and user are required."))
            self._probe_ok = False
            self.completeChanged.emit()
            return

        QApplication.setOverrideCursor(_WAIT_CURSOR)
        try:
            result = DbProbe(wiz.profile).probe()
        finally:
            QApplication.restoreOverrideCursor()
        wiz.last_probe = result

        if not result.server_reachable:
            self.lbl_status.setText(
                _bad(f"Server not reachable: {wiz.profile.host}:{wiz.profile.port}")
                + (
                    f"\n{result.error_messages[0]}" if result.error_messages else ""
                )
            )
            self._probe_ok = False
        elif not result.can_login:
            self.lbl_status.setText(
                _bad("Server reached but login failed. Check user/password.")
                + (
                    f"\n{result.error_messages[0]}" if result.error_messages else ""
                )
            )
            self._probe_ok = False
        else:
            self.lbl_status.setText(
                _ok(f"Connected (PostgreSQL {result.server_version}).")
                + "  Click Next to review capabilities."
            )
            self._probe_ok = True
        self.completeChanged.emit()


# ---------------------------------------------------------------------------
# Page 3 - Capabilities
# ---------------------------------------------------------------------------


class CapabilityPage(QWizardPage):
    """Render the latest :class:`ProbeResult` and let the user fix each gap."""

    def __init__(self, wizard: DbSetupWizard):
        super().__init__(wizard)
        self.setTitle("Database capabilities")
        self.setSubTitle(
            "Each row shows whether a required piece is in place.  Apply the "
            "missing ones; the wizard re-probes after every action."
        )

        outer = QVBoxLayout(self)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setWordWrap(True)
        outer.addWidget(self.lbl_summary)

        self.checks_box = QGroupBox("Checks")
        self.checks_layout = QVBoxLayout(self.checks_box)
        outer.addWidget(self.checks_box)

        actions = QGroupBox("Actions")
        actions_layout = QHBoxLayout(actions)
        self.btn_create_postgis = QPushButton("Enable PostGIS")
        self.btn_create_postgis.clicked.connect(self._on_create_postgis)
        self.btn_apply_migrations = QPushButton("Apply OMRAT schema migrations")
        self.btn_apply_migrations.clicked.connect(self._on_apply_migrations)
        self.btn_reprobe = QPushButton("Re-probe")
        self.btn_reprobe.clicked.connect(self._on_reprobe)
        actions_layout.addWidget(self.btn_create_postgis)
        actions_layout.addWidget(self.btn_apply_migrations)
        actions_layout.addWidget(self.btn_reprobe)
        actions_layout.addStretch(1)
        outer.addWidget(actions)

        year_box = QGroupBox("Provision year-partitioned tables (optional)")
        year_layout = QHBoxLayout(year_box)
        year_layout.addWidget(QLabel("Year:"))
        self.sb_year = QSpinBox()
        self.sb_year.setRange(1900, 2999)
        self.sb_year.setValue(2024)
        # Default Qt sizing clips the 4-digit year on Windows; give it
        # enough room for "2999" plus the up/down arrows without ellipsis.
        self.sb_year.setMinimumWidth(90)
        year_layout.addWidget(self.sb_year)
        self.btn_year = QPushButton("Create tables")
        self.btn_year.clicked.connect(self._on_year_partition)
        year_layout.addWidget(self.btn_year)
        year_layout.addStretch(1)
        outer.addWidget(year_box)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        outer.addWidget(self.log)

    # -- QWizardPage hooks ---------------------------------------------------

    def initializePage(self) -> None:
        self._render_probe()

    def isComplete(self) -> bool:
        result = self._wiz().last_probe
        return bool(result and result.ready_for_omrat)

    # -- helpers -------------------------------------------------------------

    def _wiz(self) -> DbSetupWizard:
        return self.wizard()  # type: ignore[return-value]

    def _log(self, msg: str) -> None:
        self.log.appendPlainText(msg)

    def _render_probe(self) -> None:
        result = self._wiz().last_probe
        # Clear existing check rows.
        while self.checks_layout.count():
            item = self.checks_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if result is None:
            self.lbl_summary.setText("No probe result available - go back and Test the connection.")
            self._set_action_state(False, False)
            self.completeChanged.emit()
            return

        rows = [
            (result.server_reachable, f"Server reachable ({self._wiz().profile.host}:{self._wiz().profile.port})"),
            (result.can_login, f"Login OK as {self._wiz().profile.user!r}"),
            (
                result.postgis_installed,
                f"PostGIS extension ({result.postgis_version or 'not installed'})",
            ),
            (
                result.target_schema_present,
                f"Target schema {self._wiz().profile.schema!r} exists",
            ),
            (
                result.omrat_meta_present,
                f"omrat_meta schema present (version={result.schema_version})",
            ),
        ]
        for ok, label in rows:
            self.checks_layout.addWidget(QLabel(_ok(label) if ok else _bad(label)))

        # Optional/info-only rows that don't block.
        info_rows = []
        if result.timescaledb_installed:
            info_rows.append(_info(f"TimescaleDB also detected ({result.timescaledb_version})"))
        info_rows.append(
            _info(
                f"Privileges: superuser={result.is_superuser}, "
                f"create-schema={result.can_create_schema}, "
                f"create-extension={result.can_create_extension}"
            )
        )
        for label in info_rows:
            self.checks_layout.addWidget(QLabel(label))

        self.lbl_summary.setText(
            "All required pieces are in place." if result.ready_for_omrat
            else "One or more required pieces are missing - apply the actions below."
        )

        # Action enablement reflects what's actionable from the probe result.
        self._set_action_state(
            postgis_actionable=(not result.postgis_installed) and result.is_superuser,
            migrations_actionable=(
                not result.omrat_meta_present or (result.schema_version is None)
            ) and result.can_login,
        )
        # Year-partition button needs the meta schema in place first.
        self.btn_year.setEnabled(result.omrat_meta_present)

        self.completeChanged.emit()

    def _set_action_state(self, postgis_actionable: bool, migrations_actionable: bool) -> None:
        self.btn_create_postgis.setEnabled(postgis_actionable)
        if not postgis_actionable:
            result = self._wiz().last_probe
            if result and not result.is_superuser and not result.postgis_installed:
                self.btn_create_postgis.setToolTip(
                    "PostGIS is missing but the current user is not a superuser. "
                    "Ask a DBA to run: CREATE EXTENSION postgis;"
                )
            else:
                self.btn_create_postgis.setToolTip("")
        self.btn_apply_migrations.setEnabled(migrations_actionable)

    # -- action handlers -----------------------------------------------------

    def _on_reprobe(self) -> None:
        wiz = self._wiz()
        QApplication.setOverrideCursor(_WAIT_CURSOR)
        try:
            wiz.last_probe = DbProbe(wiz.profile).probe()
        finally:
            QApplication.restoreOverrideCursor()
        self._log("Re-probed.")
        self._render_probe()

    def _on_create_postgis(self) -> None:
        import psycopg2  # local import — module already in deps via compute/database.py
        wiz = self._wiz()
        QApplication.setOverrideCursor(_WAIT_CURSOR)
        try:
            try:
                conn = psycopg2.connect(**wiz.profile.to_dsn())
            except Exception as e:
                self._log(f"PostGIS: could not connect: {e}")
                return
            try:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis")
                conn.commit()
                self._log("PostGIS extension enabled.")
            except Exception as e:
                conn.rollback()
                self._log(f"PostGIS: CREATE EXTENSION failed: {e}")
                QMessageBox.warning(
                    self,
                    "Could not enable PostGIS",
                    f"CREATE EXTENSION postgis failed:\n\n{e}\n\n"
                    "PostGIS install usually requires the postgis package on "
                    "the server host plus superuser privileges.  Ask a DBA to run:\n"
                    "    CREATE EXTENSION postgis;",
                )
            finally:
                conn.close()
        finally:
            QApplication.restoreOverrideCursor()
        self._on_reprobe()

    def _on_apply_migrations(self) -> None:
        wiz = self._wiz()
        QApplication.setOverrideCursor(_WAIT_CURSOR)
        try:
            try:
                applied = Migrator(wiz.profile).apply_pending()
            except MigrationError as e:
                self._log(f"Migrations failed: {e}")
                QMessageBox.warning(self, "Migration failed", str(e))
                return
        finally:
            QApplication.restoreOverrideCursor()
        if not applied:
            self._log("No pending migrations.")
        else:
            for m in applied:
                self._log(f"Applied V{m.version:03d} ({m.name}).")
        self._on_reprobe()

    def _on_year_partition(self) -> None:
        wiz = self._wiz()
        year = int(self.sb_year.value())
        QApplication.setOverrideCursor(_WAIT_CURSOR)
        try:
            try:
                Migrator(wiz.profile).ensure_year_partition(year)
            except MigrationError as e:
                self._log(f"ensure_year_partition({year}) failed: {e}")
                QMessageBox.warning(self, "Year-partition failed", str(e))
                return
        finally:
            QApplication.restoreOverrideCursor()
        self._log(f"Year-partitioned tables ready for {year}.")


# ---------------------------------------------------------------------------
# Page 4 - Ingest AIS
# ---------------------------------------------------------------------------


class IngestPage(QWizardPage):
    """Decode raw AIS files and write TDKC linestring segments to the DB.

    The actual work runs on a :class:`QThread` (see
    :func:`omrat_utils.handle_ais_ingest.make_worker`).  Page navigation is
    locked while a worker is running so the user cannot Finish mid-ingest.
    """

    def __init__(self, wizard: DbSetupWizard):
        super().__init__(wizard)
        self.setTitle("Ingest AIS data (optional)")
        self.setSubTitle(
            "Decode raw AIS files (NMEA / aisdb-CSV) and compress them into "
            "linestring segments using TDKC.  Skip this step if you only "
            "wanted to set up an empty database."
        )
        self._files: list[str] = []
        self._worker = None  # type: ignore[assignment]
        self._running = False

        outer = QVBoxLayout(self)

        # ---- File picker ----
        files_box = QGroupBox("Source files")
        files_layout = QVBoxLayout(files_box)
        self.list_files = QListWidget()
        self.list_files.setMaximumHeight(120)
        files_layout.addWidget(self.list_files)
        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add files...")
        btn_add.clicked.connect(self._on_add_files)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._on_clear_files)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch(1)
        files_layout.addLayout(btn_row)
        outer.addWidget(files_box)

        # ---- Compression / target settings ----
        cfg_box = QGroupBox("TDKC + target settings")
        cfg_form = QFormLayout(cfg_box)
        self.sb_min_sed = QDoubleSpinBox()
        self.sb_min_sed.setDecimals(2)
        self.sb_min_sed.setRange(0.0, 100_000.0)
        self.sb_min_sed.setSuffix(" m")
        self.sb_min_svd = QDoubleSpinBox()
        self.sb_min_svd.setDecimals(3)
        self.sb_min_svd.setRange(0.0, 1_000.0)
        self.sb_min_svd.setSuffix(" kn")
        self.sb_max_gap = QDoubleSpinBox()
        self.sb_max_gap.setDecimals(0)
        self.sb_max_gap.setRange(60.0, 86_400.0)
        self.sb_max_gap.setSingleStep(60.0)
        self.sb_max_gap.setSuffix(" s")
        self.sb_max_gap.setToolTip(
            "Split a vessel's track wherever consecutive AIS pings are "
            "further apart in time than this.  Default 3600 s (1 h) handles "
            "typical receiver dropouts; lower it for dense coastal data, "
            "raise it for offshore traffic with sparse coverage."
        )
        self.sb_speed_tol = QDoubleSpinBox()
        self.sb_speed_tol.setDecimals(2)
        self.sb_speed_tol.setRange(0.0, 5.0)
        self.sb_speed_tol.setSingleStep(0.05)
        self.sb_speed_tol.setToolTip(
            "Allowed fractional excess of the implied speed (haversine / dt) "
            "over the average reported SOG.  0.30 = split whenever the "
            "implied speed exceeds avg_sog × 1.3 + speed_floor_kn."
        )
        self.sb_speed_floor = QDoubleSpinBox()
        self.sb_speed_floor.setDecimals(2)
        self.sb_speed_floor.setRange(0.0, 50.0)
        self.sb_speed_floor.setSingleStep(0.1)
        self.sb_speed_floor.setSuffix(" kn")
        self.sb_speed_floor.setToolTip(
            "Additive slack on top of the percentage limit, so GPS jitter "
            "for slow / moored vessels doesn't trigger spurious splits."
        )
        self.sb_year = QSpinBox()
        self.sb_year.setRange(1900, 2999)
        # Sentinel: when the value sits at the minimum (1900), display
        # "— pick year —" instead of the number so the user is forced to
        # engage with this control.  Pings outside the chosen year are
        # silently dropped, so picking the wrong year ingests nothing —
        # we'd rather block a run with no year than ship that footgun.
        self.sb_year.setSpecialValueText("— pick year —")
        self.sb_year.setMinimumWidth(140)
        self.sb_year.setToolTip(
            "Selects the year-partitioned tables to populate "
            "(segments_YYYY_*, statics_YYYY, states_YYYY).  Pings whose "
            "UTC timestamp falls outside [YYYY-01-01, YYYY+1-01-01) are "
            "SKIPPED — if your input file is 2019 data and you set the "
            "year to 2026, nothing will be ingested.  Match this to the "
            "actual year inside the file."
        )
        self.le_source = QLineEdit("OMRAT")
        cfg_form.addRow("min_sed_m (SED floor)", self.sb_min_sed)
        cfg_form.addRow("min_svd_kn (SVD floor)", self.sb_min_svd)
        cfg_form.addRow("max_gap_s (track-split time gap)", self.sb_max_gap)
        cfg_form.addRow("speed_tolerance (fraction)", self.sb_speed_tol)
        cfg_form.addRow("speed_floor_kn (jitter slack)", self.sb_speed_floor)
        cfg_form.addRow("Target year", self.sb_year)
        cfg_form.addRow("Source label", self.le_source)

        # ---- Clear previous data ----
        clear_box = QGroupBox("Clear previous data (optional)")
        clear_layout = QHBoxLayout(clear_box)
        clear_layout.addWidget(QLabel("Year:"))
        self.sb_clear_year = QSpinBox()
        self.sb_clear_year.setRange(1900, 2999)
        self.sb_clear_year.setMinimumWidth(90)
        clear_layout.addWidget(self.sb_clear_year)
        self.btn_clear_year = QPushButton("Clear year data...")
        self.btn_clear_year.setToolTip(
            "Wipes every row in the year's segments, states and statics "
            "tables, plus the matching watermark entries.  Useful when "
            "re-ingesting the same year with different settings.  "
            "Tables are kept (only data is cleared)."
        )
        self.btn_clear_year.clicked.connect(self._on_clear_year)
        clear_layout.addWidget(self.btn_clear_year)
        clear_layout.addStretch(1)
        outer.addWidget(clear_box)

        self.cb_incremental = QCheckBox(
            "Incremental: skip pings already represented in the watermark"
        )
        self.cb_incremental.setChecked(True)
        self.cb_incremental.setToolTip(
            "Reads omrat_meta.segment_watermark per MMSI and ignores AIS pings "
            "whose timestamp is at-or-before the last one already ingested.  "
            "Uncheck only if you've wiped the segments tables and want to "
            "force a full re-ingest."
        )
        cfg_form.addRow("", self.cb_incremental)
        outer.addWidget(cfg_box)

        # ---- Run / progress ----
        warn = QLabel(
            "<i>Ingestion can take minutes to hours on real datasets.  "
            "The status box below updates as files are read; you can hit "
            "Cancel to stop cleanly — the per-MMSI watermark means a "
            "follow-up run with the same files resumes where you stopped.</i>"
        )
        warn.setWordWrap(True)
        outer.addWidget(warn)

        run_row = QHBoxLayout()
        self.btn_run = QPushButton("Run ingestion")
        self.btn_run.clicked.connect(self._on_run)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        run_row.addWidget(self.btn_run)
        run_row.addWidget(self.btn_cancel)
        run_row.addStretch(1)
        outer.addLayout(run_row)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        outer.addWidget(self.log)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        outer.addWidget(self.lbl_status)

    # -- QWizardPage hooks ---------------------------------------------------

    def initializePage(self) -> None:
        wiz = self._wiz()
        s = wiz.ingest_settings
        self.sb_min_sed.setValue(float(s.min_sed_m))
        self.sb_min_svd.setValue(float(s.min_svd_kn))
        self.sb_max_gap.setValue(float(s.max_gap_s))
        self.sb_speed_tol.setValue(float(s.speed_tolerance))
        self.sb_speed_floor.setValue(float(s.speed_floor_kn))
        # Deliberately do NOT pre-fill Target year.  Auto-defaulting to
        # the current year was a real footgun — users would accept the
        # default, run, and silently get zero segments because their
        # input file's pings fell in a different year.  The sentinel
        # value (1900) shows "— pick year —" until the user changes it,
        # and ``_on_run`` / ``_on_clear_year`` refuse the sentinel.
        if self.sb_clear_year.value() < 2000:
            # Mirror the (sentinel) ingest year so the clear-year picker
            # also starts in the "must pick" state.
            self.sb_clear_year.setValue(self.sb_year.value())

    def isComplete(self) -> bool:
        # Always allow Next: ingestion is optional.  But block while a worker
        # is running.
        return not self._running

    def validatePage(self) -> bool:
        # Persist whatever the user typed into the settings, even if they
        # skipped running the worker.  Saved values become the next-time
        # defaults.
        self._capture_into_settings()
        self._wiz().ingest_settings.to_qsettings()
        return True

    # -- helpers -------------------------------------------------------------

    def _wiz(self) -> DbSetupWizard:
        return self.wizard()  # type: ignore[return-value]

    def _capture_into_settings(self) -> None:
        s = self._wiz().ingest_settings
        s.min_sed_m = float(self.sb_min_sed.value())
        s.min_svd_kn = float(self.sb_min_svd.value())
        s.max_gap_s = float(self.sb_max_gap.value())
        s.speed_tolerance = float(self.sb_speed_tol.value())
        s.speed_floor_kn = float(self.sb_speed_floor.value())

    def _log(self, msg: str) -> None:
        self.log.appendPlainText(msg)

    # -- file picker ---------------------------------------------------------

    def _on_add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select AIS files",
            "",
            "AIS files (*.nm4 *.nmea *.csv *.csv.gz *.gz);;All files (*.*)",
        )
        for f in files:
            if f and f not in self._files:
                self._files.append(f)
                self.list_files.addItem(f)

    def _on_clear_files(self) -> None:
        self._files.clear()
        self.list_files.clear()

    # -- run -----------------------------------------------------------------

    def _on_run(self) -> None:
        if self._running:
            return
        if not self._files:
            QMessageBox.information(
                self, "No files selected", "Add at least one AIS file before running."
            )
            return
        wiz = self._wiz()
        if not wiz.last_probe or not wiz.last_probe.ready_for_omrat:
            QMessageBox.warning(
                self,
                "Database not ready",
                "Go back to the Capabilities page and apply the OMRAT migrations "
                "before ingesting.",
            )
            return
        year = int(self.sb_year.value())
        if year < 2000:
            QMessageBox.warning(
                self,
                "Pick a target year",
                "Target year hasn't been set.  Pick the year that matches "
                "the timestamps in your AIS files — pings outside that "
                "year are silently dropped, so the wrong year ingests "
                "nothing.",
            )
            return
        # Confirmation: ingestion is long-running and easy to start by
        # accident, so make the user explicitly OK it.
        confirm = QMessageBox.question(
            self,
            "Start ingestion?",
            f"Decode and ingest <b>{len(self._files)} file(s)</b> into "
            f"<code>segments_{year}</code> tables?<br><br>"
            "This can take <b>minutes to hours</b> on real datasets — large "
            "Marine-Cadastre files take ~1-3 minutes per pass to read, and "
            "are read twice (once for static info, once for kinematics) "
            "before any segments are written.<br><br>"
            "You can hit <b>Cancel</b> at any time to stop cleanly; "
            "committed batches are preserved and a follow-up run resumes "
            "where you stopped.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._capture_into_settings()
        self._wiz().ingest_settings.to_qsettings()

        # Lazy import keeps Qt-thread-creation cost out of the wizard's
        # opening path, and keeps the headless pipeline test-importable.
        from omrat_utils.handle_ais_ingest import make_worker

        worker = make_worker(
            profile=wiz.profile,
            settings=wiz.ingest_settings,
            files=list(self._files),
            year=year,
            source_tag=self.le_source.text().strip() or "OMRAT",
            incremental=self.cb_incremental.isChecked(),
        )
        self._worker = worker
        worker.message.connect(self._log)
        worker.finished_with_result.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        self._running = True
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_clear_year.setEnabled(False)
        self.lbl_status.setText("Running ingestion…")
        self.completeChanged.emit()
        worker.start()

    def _on_cancel(self) -> None:
        """Request a clean stop of the running worker.

        Cooperative: the pipeline checks the cancellation flag before each
        track, so the worker may take a few seconds to exit (it finishes
        the file currently being read, then bails on the next track).
        """
        if not self._running or self._worker is None:
            return
        self._log("Cancellation requested — finishing current file...")
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("Cancelling… (waiting for current file to finish)")
        try:
            self._worker.cancel()
        except Exception as e:
            self._log(f"  cancel failed: {e}")

    def _on_finished(self, result) -> None:
        self._running = False
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_clear_year.setEnabled(True)
        self.lbl_status.setText(
            _ok(result.summary())
            if not result.errors
            else _bad(result.summary())
        )
        self._log(result.summary())
        for err in result.errors[:10]:
            self._log(f"  error: {err}")
        if len(result.errors) > 10:
            self._log(f"  ... and {len(result.errors) - 10} more errors")
        self.completeChanged.emit()

    def _on_failed(self, msg: str) -> None:
        self._running = False
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_clear_year.setEnabled(True)
        self.lbl_status.setText(_bad(f"Ingestion failed: {msg}"))
        self.completeChanged.emit()

    # -- clear-year-data -----------------------------------------------------

    def _on_clear_year(self) -> None:
        """Wipe segments / states / statics + watermarks for one year.

        Two-stage confirmation: first a normal Yes/No dialog showing the
        row counts about to be deleted, then a typed-year confirmation
        for the actual destructive action.  Skipped silently if the
        targeted year tables don't exist.
        """
        if self._running:
            return
        wiz = self._wiz()
        if not wiz.last_probe or not wiz.last_probe.ready_for_omrat:
            QMessageBox.warning(
                self,
                "Database not ready",
                "The OMRAT schema must be set up before data can be cleared.",
            )
            return
        year = int(self.sb_clear_year.value())
        if year < 2000:
            QMessageBox.warning(
                self,
                "Pick a year to clear",
                "Pick the year of data you want to wipe (this is "
                "deliberately not pre-filled to avoid accidental clears).",
            )
            return
        try:
            counts = Migrator(wiz.profile).count_year_data(year)
        except MigrationError as e:
            QMessageBox.warning(self, "Could not count rows", str(e))
            return
        if not counts or all(v == 0 for v in counts.values()):
            QMessageBox.information(
                self,
                "Nothing to clear",
                f"No data found for year {year} (tables either empty or "
                "not provisioned).",
            )
            return
        rows_html = "<br>".join(
            f"&nbsp;&nbsp;<code>{tbl}</code>: <b>{count:,}</b> rows"
            for tbl, count in counts.items()
        )
        confirm = QMessageBox.warning(
            self,
            f"Clear all {year} data?",
            f"This will <b>delete every row</b> in the {year} tables:<br><br>"
            f"{rows_html}<br><br>"
            "The schema itself is preserved, only data is wiped.  "
            "<b>This cannot be undone.</b><br><br>Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        # Second-stage confirmation: type the year.
        from qgis.PyQt.QtWidgets import QInputDialog
        typed, ok = QInputDialog.getText(
            self,
            "Confirm year",
            f"Type <b>{year}</b> to confirm:",
        )
        if not ok or typed.strip() != str(year):
            self._log("Clear cancelled (year confirmation didn't match).")
            return
        QApplication.setOverrideCursor(_WAIT_CURSOR)
        try:
            try:
                Migrator(wiz.profile).truncate_year(year)
            except MigrationError as e:
                self._log(f"truncate_year({year}) failed: {e}")
                QMessageBox.warning(self, "Clear failed", str(e))
                return
        finally:
            QApplication.restoreOverrideCursor()
        self._log(f"Cleared all {year} data.")
        QMessageBox.information(
            self,
            "Cleared",
            f"All {year} data has been cleared.  The tables remain "
            "and are ready for a fresh ingestion run.",
        )


# ---------------------------------------------------------------------------
# Page 5 - Done
# ---------------------------------------------------------------------------


class DonePage(QWizardPage):
    """Persist the profile and show a final summary."""

    def __init__(self, wizard: DbSetupWizard):
        super().__init__(wizard)
        self.setTitle("Setup complete")
        self.setSubTitle("Review the settings, then click Finish to save.")

        layout = QVBoxLayout(self)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.RichText if hasattr(Qt, "RichText") else Qt.TextFormat.RichText)
        layout.addWidget(self.summary)

        self.cb_save = QCheckBox("Save these connection settings as the default profile")
        self.cb_save.setChecked(True)
        layout.addWidget(self.cb_save)

        layout.addStretch(1)

    def initializePage(self) -> None:
        wiz = self._wiz()
        p = wiz.profile
        result = wiz.last_probe
        ready = "Yes" if (result and result.ready_for_omrat) else "Not yet"
        self.summary.setText(
            "<b>Connection</b><br>"
            f"Host: {p.host}:{p.port}<br>"
            f"Database: {p.database}<br>"
            f"User: {p.user}<br>"
            f"Schema: {p.schema}<br>"
            f"SSL mode: {p.sslmode}<br>"
            "<br>"
            f"<b>Ready for OMRAT:</b> {ready}<br>"
            "<br>"
            "Saving the profile updates both the new <code>omrat/db_profiles/default/*</code> "
            "keys and the legacy flat keys read by the AIS connection dialog, so existing "
            "AIS queries pick up the same credentials immediately."
        )

    def validatePage(self) -> bool:
        if self.cb_save.isChecked():
            self._wiz().profile.to_qsettings()
        return True

    def _wiz(self) -> DbSetupWizard:
        return self.wizard()  # type: ignore[return-value]
