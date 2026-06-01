from dataclasses import dataclass, field
from operator import xor
import json
from functools import partial
from typing import Optional, Any, TYPE_CHECKING, cast, Union
if TYPE_CHECKING:
    from omrat import OMRAT, OMRATMainWidget

import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import numpy as np
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QSpinBox, QDoubleSpinBox, QLineEdit, QCheckBox
from scipy import stats

from ui.traffic_data_widget import TrafficDataWidget
from geometries import isint


WidgetType = Union[QLineEdit, QSpinBox, QDoubleSpinBox]

SCALING_VAR = 'Scaling (%)'
SCALING_DEFAULT = 100.0


def _default_traffic_scaling() -> dict[str, Any]:
    """Return the empty / new-project traffic-scaling state.

    Used by both ``omrat.OMRAT.__init__`` (live state) and the gather /
    populate pair (round-trip via ``.omrat``).
    """
    return {'global_percent': SCALING_DEFAULT, 'follow_global': []}


class Traffic:
    def __init__(self, omrat: "OMRAT", dw:"OMRATMainWidget"):
        self.dw = dw
        self.omrat = omrat
        self.traffic_data: dict[str, dict[str, dict[str, Any]]] = self.omrat.traffic_data
        self.c_seg = "1"
        self.c_di = ""
        self.current_table = []
        # ``Scaling (%)`` is a per-cell frequency multiplier (default 100).
        # It lives in the same nested-dict structure as the other vars so
        # save / load / table editing all just work; compute multiplies Q
        # by Scaling / 100 at the top of ``CalculationTask.run``.
        self.variables = [
            'Frequency (ships/year)', 'Speed (knots)', 'Draught (meters)',
            'Ship heights (meters)', 'Ship Beam (meters)', SCALING_VAR,
        ]
        # Per-variable cell-default used by ``create_empty_dict``.  Anything
        # not listed here falls back to an empty list (= "no AIS observations
        # yet").  Scaling defaults to 100 % so unchanged projects are no-ops.
        self._var_cell_defaults: dict[str, Any] = {
            'Frequency (ships/year)': 0,
            SCALING_VAR: SCALING_DEFAULT,
        }
        # Checkboxes (one per ship-type row) are populated in
        # ``populate_scaling_checkboxes``.  We hold strong references here so
        # GC + Qt's parent-chain don't drop them prematurely.
        self._follow_global_checkboxes: list[QCheckBox] = []
        self._scaling_cell_signals: list[Any] = []
        self.last_var = 'Frequency (ships/year)'
        self.run_update = False
        self.dw.cbSelectType.currentIndexChanged.connect(partial(self.update_traffic_tbl, 'type'))
        self.dw.cbTrafficSelectSeg.currentIndexChanged.connect(partial(self.update_traffic_tbl, 'segment'))
        self.dw.cbTrafficDirectionSelect.currentIndexChanged.connect(partial(self.update_traffic_tbl, 'dir'))
        self._connect_scaling_controls()
        self.set_table_headings()
        self.run_update = True
        
    def fill_cbTrafficSelectSeg(self):
        """Sets the segment names in cbTrafficSelectSeg"""
        self.dw.cbTrafficSelectSeg.clear()
        nrs = self.dw.twRouteList.rowCount()
        for i in range(nrs):
            self.dw.cbTrafficSelectSeg.addItem(self.dw.twRouteList.item(i, 0).text())
        self.c_seg = self.dw.cbTrafficSelectSeg.currentText()

    def set_table_headings(self):
        """Sets the column and row names of the table"""
        types: list[str] = []
        for i in range(self.omrat.ship_cat.scw.cvTypes.rowCount()):
            it = self.omrat.ship_cat.scw.cvTypes.item(i, 0)
            text = it.text() if it is not None else ""
            types.append(text)
        sizes: list[str] = []
        for i in range(self.omrat.ship_cat.scw.twLengths.rowCount()):
            it1 = self.omrat.ship_cat.scw.twLengths.item(i, 0)
            text1 = it1.text() if it1 is not None else ""
            it2 = self.omrat.ship_cat.scw.twLengths.item(i, 1)
            text2 = it2.text() if it2 is not None else ""
            sizes.append(f'{text1} - {text2}')
        self.dw.twTrafficData.setColumnCount(len(sizes))
        self.dw.twTrafficData.setHorizontalHeaderLabels(sizes)
        self.dw.twTrafficData.setRowCount(len(types))
        self.dw.twTrafficData.setVerticalHeaderLabels(types)
        # Ensure the vertical header is visible and wide enough to display
        # the ship-type labels.  The main-widget styling hides vertical
        # headers on other tables; this force-shows it specifically for the
        # traffic-data table regardless of that styling path.
        vh = self.dw.twTrafficData.verticalHeader()
        vh.setVisible(True)
        if types:
            # Size the header to fit the widest label + a little padding.
            # Guard with try/except so tests that pass a MagicMock widget
            # (e.g. tests/test_ais_data_retrival.py) don't crash on
            # `max()` over mocked fontMetrics values.
            try:
                fm = self.dw.twTrafficData.fontMetrics()
                widths = [int(fm.horizontalAdvance(t)) for t in types]
                vh.setDefaultSectionSize(28)
                vh.setMinimumWidth(max(widths) + 16)
            except (TypeError, ValueError):
                # Mocked widget or fontMetrics not usable; skip sizing.
                pass
        for row in range(len(types)):
            for col in range(len(sizes)):
                item = QSpinBox()
                item.setMaximum(100000)
                self.dw.twTrafficData.setCellWidget(row, col, item)
        # Keep the per-type "follow global" checkboxes + the follow_global
        # bool list in sync with the current ship-type count.
        self.populate_scaling_checkboxes()

    def create_empty_dict(self, s_key:str, dirs:list[str]):
        """Creates an empty dict for the segment with all types"""
        self.traffic_data[s_key] = {}
        rows = self.dw.twTrafficData.rowCount()
        cols = self.dw.twTrafficData.columnCount()

        for di in dirs:
            self.traffic_data[s_key][di] = {}
            for key in self.variables:
                default = self._var_cell_defaults.get(key, [])
                self.traffic_data[s_key][di][key] = []
                for _ in range(rows):
                    line: list[Any] = []
                    for _ in range(cols):
                        # Each cell needs its own object so that AIS code
                        # appending into Speed/Beam/etc. lists does not
                        # accidentally write through to other cells.
                        line.append(default if not isinstance(default, list) else [])
                    self.traffic_data[s_key][di][key].append(line)
            
    def update_traffic_tbl(self, caller:str):
        """Updates Traffic data table with the data from traffic_data,
        using the correct type and segment"""
        if self.run_update:
            self.save()
        if caller == 'segment':
            self.c_seg = self.dw.cbTrafficSelectSeg.currentText()
            self.update_direction_select()
        rows = self.dw.twTrafficData.rowCount()
        cols = self.dw.twTrafficData.columnCount()
        self.last_var: str = self.dw.cbSelectType.currentText()
        self.c_di = self.dw.cbTrafficDirectionSelect.currentText()
        if any([self.c_seg== "", self.c_di== "", self.last_var== ""]):
            return
        # Make sure the Scaling matrix exists for the (seg, dir) we are
        # about to display.  Legacy projects and freshly-imported IWRAP
        # files may not have it yet.
        self.ensure_scaling_present()
        is_scaling_view = self.last_var == SCALING_VAR
        # Drop any cellChanged-style signals from the previous render so
        # they don't fire while we rebuild widgets below.
        self._disconnect_scaling_cell_signals()
        for row in range(rows):
            for col in range(cols):
                val:float|int|str = self.traffic_data[self.c_seg][self.c_di][self.last_var][row][col]

                if is_scaling_view:
                    # Scaling cells are always percent values rendered as
                    # a wide-range double spinbox -- ``isint(100.0)`` is
                    # True so the generic dispatch below would otherwise
                    # truncate fractional percentages on edit.
                    item = QDoubleSpinBox()
                    item.setRange(0.0, 100000.0)
                    item.setSuffix(' %')
                    item.setDecimals(1)
                    try:
                        v = float(val) if val != '' else SCALING_DEFAULT
                    except (TypeError, ValueError):
                        v = SCALING_DEFAULT
                    item.setValue(v)
                    # Manual edit -> the row stops following the global
                    # spinbox.  Connect after setValue so the initial
                    # render doesn't auto-untick everything.
                    item.valueChanged.connect(
                        partial(self._on_scaling_cell_edited, row)
                    )
                    self._scaling_cell_signals.append((item, row))
                elif val == '':
                    item = QSpinBox()
                    val = 0
                    item.setValue(val)
                elif val == np.inf:
                    item = QSpinBox()
                    item.setEnabled(False)
                    val = 0
                    item.setValue(val)
                elif isint(val):
                    item = QSpinBox()
                    item.setMaximum(100000)
                    val = int(val)
                    item.setValue(val)
                else:
                    item = QDoubleSpinBox()
                    val = float(val)
                    item.setValue(val)
                self.dw.twTrafficData.setCellWidget(row, col, item)
        
    def update_direction_select(self):
        self.run_update = False
        self.dw.cbTrafficDirectionSelect.clear()
        if len(self.traffic_data) == 0 or self.c_seg == '':
            return
        for key in self.traffic_data[self.c_seg].keys():
            self.dw.cbTrafficDirectionSelect.addItem(key)
        self.c_di = self.dw.cbTrafficDirectionSelect.currentText()
        self.run_update = True
    
    def save(self):
        """Saves the previous "table" in traffic_data"""
        if any([self.c_seg == "", self.c_di == "", self.last_var == ""]):
            return
        rows = self.dw.twTrafficData.rowCount()
        cols = self.dw.twTrafficData.columnCount()
        typ = self.last_var
        for row in range(rows):
            for col in range(cols):
                val = self.dw.twTrafficData.cellWidget(row, col)
                self.traffic_data[self.c_seg][self.c_di][typ][row][col] = val.value()
                    
    def commit_changes(self):
        """Copy the output from the traffic data within this module to omrats traffic_data"""
        self.save()
        self.omrat.traffic_data = self.traffic_data

    # --------------------------------------------------------------- scaling
    # The "easy option" scaling lets a user bump every Frequency cell
    # by a global percentage, with per-ship-type opt-outs.  Storage is
    # in ``traffic_data[seg][dir]['Scaling (%)'][type_idx][len_idx]``;
    # ``omrat.traffic_scaling`` holds ``{global_percent, follow_global}``
    # at the project level.  Compute multiplies Q by Scaling / 100 in
    # ``compute.data_preparation.apply_traffic_scaling``.

    def _connect_scaling_controls(self) -> None:
        """Connect the global spinbox + reset button + collapse toggle.

        These widgets live on the main OMRAT widget (added in
        ``omrat_base.ui``).  Tests / older UIs that don't have them just
        silently skip; the per-cell editing still works.
        """
        mw = getattr(self.omrat, 'main_widget', None) or self.dw
        sb = getattr(mw, 'dsbGlobalTrafficScaling', None)
        if sb is not None:
            try:
                sb.editingFinished.connect(self._on_global_scaling_changed)
            except (TypeError, AttributeError):
                pass
        btn = getattr(mw, 'pbResetTrafficScaling', None)
        if btn is not None:
            try:
                btn.clicked.connect(self._on_reset_scaling_clicked)
            except (TypeError, AttributeError):
                pass
        # Checkable groupbox -- when the user unticks the title, hide the
        # inner content widget so the layout reclaims the space (Qt's
        # default behaviour only disables children).  Default is
        # *unchecked* (collapsed) so the matrix gets full width on first
        # open; the user expands it only when they need to tweak
        # scaling.
        gb = getattr(mw, 'gbScalingControls', None)
        content = getattr(mw, 'wScalingContent', None)
        if gb is not None and content is not None:
            try:
                gb.toggled.connect(content.setVisible)
                # Force the initial visibility to match the groupbox
                # checked-state so the .ui default and the runtime
                # state can't drift apart.
                content.setVisible(bool(gb.isChecked()))
            except (TypeError, AttributeError):
                pass

    def _project_scaling(self) -> dict[str, Any]:
        """Return (and lazy-create) the live ``omrat.traffic_scaling`` dict."""
        scaling = getattr(self.omrat, 'traffic_scaling', None)
        if not isinstance(scaling, dict):
            scaling = _default_traffic_scaling()
            self.omrat.traffic_scaling = scaling
        scaling.setdefault('global_percent', SCALING_DEFAULT)
        scaling.setdefault('follow_global', [])
        return scaling

    def _disconnect_scaling_cell_signals(self) -> None:
        """Drop the cell ``valueChanged`` -> auto-untick connections.

        Qt will GC the widget eventually, but we connect on every render
        so an explicit detach keeps the signal load bounded.
        """
        for item, _row in self._scaling_cell_signals:
            try:
                item.valueChanged.disconnect()
            except (TypeError, RuntimeError):
                pass
        self._scaling_cell_signals = []

    def ensure_scaling_present(
        self, traffic_data: Optional[dict] = None,
    ) -> None:
        """Make sure every ``(seg, dir)`` has a default-100 Scaling matrix.

        Idempotent.  Used on load (legacy projects), after AIS refresh
        and after IWRAP import to seed the column without touching any
        existing user values.
        """
        td = self.traffic_data if traffic_data is None else traffic_data
        if not isinstance(td, dict):
            return
        rows = self.dw.twTrafficData.rowCount()
        cols = self.dw.twTrafficData.columnCount()
        for _seg, dirs in td.items():
            if not isinstance(dirs, dict):
                continue
            for _di, var in dirs.items():
                if not isinstance(var, dict):
                    continue
                existing = var.get(SCALING_VAR)
                freq = var.get('Frequency (ships/year)', [])
                # Shape from the existing Frequency matrix so we cover
                # legacy files where the table dims still reflect the
                # saved project (not the current UI).
                tgt_rows = len(freq) if freq else rows
                tgt_cols = len(freq[0]) if freq and hasattr(freq[0], '__len__') else cols
                if not isinstance(existing, list) or len(existing) != tgt_rows:
                    var[SCALING_VAR] = [
                        [SCALING_DEFAULT] * tgt_cols for _ in range(tgt_rows)
                    ]
                    continue
                # Right-shape rows: pad / truncate per row.
                for r in range(tgt_rows):
                    if r >= len(existing) or not isinstance(existing[r], list):
                        existing[r] = [SCALING_DEFAULT] * tgt_cols
                        continue
                    row_list = existing[r]
                    if len(row_list) < tgt_cols:
                        row_list.extend([SCALING_DEFAULT] * (tgt_cols - len(row_list)))
                    elif len(row_list) > tgt_cols:
                        del row_list[tgt_cols:]

    def ensure_follow_global(self, n_types: Optional[int] = None) -> None:
        """Resize the project-level ``follow_global`` bool list to match types.

        New rows default to True ("follow the global"); shrinking just
        truncates -- losing the unticked state for removed types is
        acceptable because the ship-type ordering is itself unstable
        across category edits.
        """
        scaling = self._project_scaling()
        if n_types is None:
            try:
                n_types = self.dw.twTrafficData.rowCount()
            except (TypeError, AttributeError):
                n_types = 0
        try:
            n_types = int(n_types)
        except (TypeError, ValueError):
            return  # Mocked widget in tests -- skip rather than crash.
        flags = scaling.get('follow_global', [])
        if not isinstance(flags, list):
            flags = []
        if len(flags) < n_types:
            flags.extend([True] * (n_types - len(flags)))
        elif len(flags) > n_types:
            flags = flags[:n_types]
        scaling['follow_global'] = flags

    def populate_scaling_checkboxes(self) -> None:
        """Rebuild the per-ship-type "follow global" checkbox column.

        Lives in ``vlFollowGlobalCheckboxes`` (a ``QVBoxLayout`` declared
        in ``omrat_base.ui``).  When the layout is missing -- e.g. in
        the standalone traffic-data dialog used by tests -- we silently
        skip and the project-level ``follow_global`` flags drive the
        broadcast logic instead.
        """
        mw = getattr(self.omrat, 'main_widget', None) or self.dw
        layout = getattr(mw, 'vlFollowGlobalCheckboxes', None)
        # Drop old checkbox widgets.
        for cb in self._follow_global_checkboxes:
            try:
                cb.toggled.disconnect()
            except (TypeError, RuntimeError):
                pass
            cb.setParent(None)
        self._follow_global_checkboxes = []
        if layout is None:
            return
        try:
            n_types = int(self.dw.twTrafficData.rowCount())
        except (TypeError, ValueError, AttributeError):
            return  # Mocked widget -- skip checkbox build.
        self.ensure_follow_global(n_types)
        flags = self._project_scaling()['follow_global']
        for i in range(n_types):
            label = ''
            try:
                hi = self.dw.twTrafficData.verticalHeaderItem(i)
                if hi is not None:
                    raw = hi.text()
                    label = raw if isinstance(raw, str) else ''
            except (TypeError, AttributeError):
                label = ''
            try:
                cb = QCheckBox(label or f'Type {i}')
            except TypeError:
                # Mocked widgets feed non-string labels through here -- just
                # skip; the project-level ``follow_global`` flags still
                # drive the broadcast logic without a UI.
                return
            cb.setChecked(bool(flags[i]) if i < len(flags) else True)
            cb.toggled.connect(partial(self._on_follow_global_toggled, i))
            try:
                layout.addWidget(cb)
            except (TypeError, AttributeError):
                pass
            self._follow_global_checkboxes.append(cb)

    def _broadcast_scaling_to_row(self, type_idx: int, value: float) -> None:
        """Write ``value`` into every cell of row ``type_idx`` across all (seg, dir)."""
        for _seg, dirs in self.traffic_data.items():
            if not isinstance(dirs, dict):
                continue
            for _di, var in dirs.items():
                if not isinstance(var, dict):
                    continue
                matrix = var.get(SCALING_VAR)
                if not isinstance(matrix, list) or type_idx >= len(matrix):
                    continue
                row = matrix[type_idx]
                if not isinstance(row, list):
                    continue
                for c in range(len(row)):
                    row[c] = value

    def apply_global_scaling(
        self, value: Optional[float] = None, refresh_table: bool = True,
    ) -> None:
        """Push the global scaling value into every "follow global" row.

        When ``value`` is None we read the current spinbox; otherwise the
        caller's value wins (used by the reset button + tests).
        """
        scaling = self._project_scaling()
        mw = getattr(self.omrat, 'main_widget', None) or self.dw
        sb = getattr(mw, 'dsbGlobalTrafficScaling', None)
        if value is None and sb is not None:
            try:
                value = float(sb.value())
            except (TypeError, AttributeError):
                value = scaling.get('global_percent', SCALING_DEFAULT)
        if value is None:
            value = scaling.get('global_percent', SCALING_DEFAULT)
        value = float(value)
        scaling['global_percent'] = value
        if sb is not None and abs(float(sb.value()) - value) > 1e-9:
            try:
                sb.blockSignals(True)
                sb.setValue(value)
            finally:
                sb.blockSignals(False)
        self.ensure_scaling_present()
        flags = scaling['follow_global']
        for i, follow in enumerate(flags):
            if not follow:
                continue
            self._broadcast_scaling_to_row(i, value)
        if refresh_table and self.last_var == SCALING_VAR:
            # Re-render so the visible spinboxes pick up the broadcast.
            prev = self.run_update
            self.run_update = False
            try:
                self.update_traffic_tbl('type')
            finally:
                self.run_update = prev

    def _on_global_scaling_changed(self) -> None:
        self.apply_global_scaling(value=None, refresh_table=True)

    def _on_reset_scaling_clicked(self) -> None:
        """Reset all scaling to 100% and re-tick every type."""
        scaling = self._project_scaling()
        scaling['global_percent'] = SCALING_DEFAULT
        flags = scaling.setdefault('follow_global', [])
        for i in range(len(flags)):
            flags[i] = True
        for cb in self._follow_global_checkboxes:
            cb.blockSignals(True)
            try:
                cb.setChecked(True)
            finally:
                cb.blockSignals(False)
        self.apply_global_scaling(value=SCALING_DEFAULT, refresh_table=True)

    def _on_follow_global_toggled(self, type_idx: int, ticked: bool) -> None:
        scaling = self._project_scaling()
        self.ensure_follow_global(self.dw.twTrafficData.rowCount())
        flags = scaling['follow_global']
        if type_idx < len(flags):
            flags[type_idx] = bool(ticked)
        if ticked:
            # Newly ticked -> broadcast the current global into this row.
            self._broadcast_scaling_to_row(
                type_idx, float(scaling.get('global_percent', SCALING_DEFAULT)),
            )
            if self.last_var == SCALING_VAR:
                prev = self.run_update
                self.run_update = False
                try:
                    self.update_traffic_tbl('type')
                finally:
                    self.run_update = prev

    def _on_scaling_cell_edited(self, type_idx: int, _value: float) -> None:
        """Auto-untick a row when the user types a value into one of its cells."""
        scaling = self._project_scaling()
        flags = scaling.setdefault('follow_global', [])
        if type_idx >= len(flags):
            return
        if not flags[type_idx]:
            return  # Already untracked.
        flags[type_idx] = False
        if type_idx < len(self._follow_global_checkboxes):
            cb = self._follow_global_checkboxes[type_idx]
            cb.blockSignals(True)
            try:
                cb.setChecked(False)
            finally:
                cb.blockSignals(False)

    def unload(self):
        """Cleanup resources and disconnect signals."""
        # Clear traffic data
        self.traffic_data.clear()
        try:
            self.dw.cbSelectType.currentIndexChanged.disconnect()
            self.dw.cbTrafficSelectSeg.currentIndexChanged.disconnect()
            self.dw.cbTrafficDirectionSelect.currentIndexChanged.disconnect()
        except TypeError:
            pass
        self._disconnect_scaling_cell_signals()
        for cb in self._follow_global_checkboxes:
            try:
                cb.toggled.disconnect()
            except (TypeError, RuntimeError):
                pass
        self._follow_global_checkboxes = []
        mw = getattr(self.omrat, 'main_widget', None) or self.dw
        for name, sig_name in (
            ('dsbGlobalTrafficScaling', 'editingFinished'),
            ('pbResetTrafficScaling', 'clicked'),
            ('gbScalingControls', 'toggled'),
        ):
            widget = getattr(mw, name, None)
            if widget is None:
                continue
            try:
                getattr(widget, sig_name).disconnect()
            except (TypeError, RuntimeError, AttributeError):
                pass
        # Remove reference to TrafficDataWidget
        print("Traffic resources cleaned up.")
