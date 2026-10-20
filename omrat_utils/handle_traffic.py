from functools import partial
from typing import Optional, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from omrat import OMRAT, OMRATMainWidget

import matplotlib as mpl
mpl.use('Qt5Agg')
import numpy as np  # noqa: E402
from qgis.PyQt.QtCore import QEvent, QLocale, QObject, Qt  # noqa: E402
from qgis.PyQt.QtGui import QColor, QKeySequence  # noqa: E402
from qgis.PyQt.QtWidgets import (  # noqa: E402
    QAbstractItemView, QApplication, QCheckBox, QMenu, QTableWidgetItem,
)

from omrat_utils.table_clipboard import (  # noqa: E402
    PastePlan, format_number, parse_number, parse_tsv, plan_paste, strip_headers, to_tsv,
)
from omrat_utils.traffic_links import status_suffix  # noqa: E402


FREQ_VAR = 'Frequency (ships/year)'
SCALING_VAR = 'Scaling (%)'
SCALING_DEFAULT = 100.0

_NO_OBSERVATION_TIP = "No AIS observation for this cell. Type a value to set one."


class _TableKeyFilter(QObject):
    """Excel-style keys on the traffic table: Ctrl+C / Ctrl+V / Delete /
    Ctrl+A.

    QGIS binds the same keys in its main window (Copy / Paste / Delete
    Selected / Select All *features*), and a window shortcut fires before
    the key reaches the focused table -- so the table claims them on
    ``ShortcutOverride`` first.  Only seen while no cell is being edited;
    the cell editor keeps its own copy / paste / delete.
    """

    _COPY_PASTE_ALL = (
        QKeySequence.StandardKey.Copy, QKeySequence.StandardKey.Paste, QKeySequence.StandardKey.SelectAll,
    )

    def __init__(self, traffic: "Traffic"):
        super().__init__()
        self._traffic = traffic

    def _ours(self, event) -> bool:
        return any(event.matches(k) for k in self._COPY_PASTE_ALL) or event.key() == Qt.Key.Key_Delete

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
        if event.type() == QEvent.Type.ShortcutOverride and self._ours(event):
            event.accept()      # deliver as a KeyPress to the table, not to QGIS
            return True
        if event.type() == QEvent.Type.KeyPress:
            if event.matches(QKeySequence.StandardKey.Copy):
                self._traffic.copy_selection()
                return True
            if event.matches(QKeySequence.StandardKey.Paste):
                self._traffic.paste_clipboard()
                return True
            if event.key() == Qt.Key.Key_Delete:
                self._traffic.clear_selection()
                return True
        return False            # Ctrl+A: the table's own select-all


def _default_traffic_scaling() -> dict[str, Any]:
    """Return the empty / new-project traffic-scaling state.

    Used by both ``omrat.OMRAT.__init__`` (live state) and the gather /
    populate pair (round-trip via ``.omrat``).
    """
    return {'global_percent': SCALING_DEFAULT, 'follow_global': []}


class Traffic:
    def __init__(self, omrat: "OMRAT", dw: "OMRATMainWidget"):
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
        # Data dimensions (excluding the summary row/column added to twTrafficData).
        self._data_rows = 0
        self._data_cols = 0
        # Checkboxes (one per ship-type row) are populated in
        # ``populate_scaling_checkboxes``.  We hold strong references here so
        # GC + Qt's parent-chain don't drop them prematurely.
        self._follow_global_checkboxes: list[QCheckBox] = []
        # The table holds plain text items (selectable, copyable to Excel).
        # ``_shown[row][col] = (stored value, rendered text)`` for the
        # matrix on screen: a cell is only written back when its text
        # differs, so viewing never rounds or overwrites data (inf stays
        # inf, 1234.5 stays 1234.5).  ``_filling`` mutes ``itemChanged``
        # while we write the cells ourselves; ``_bulk`` defers the totals
        # during a paste.
        self._shown: list[list[tuple[Any, str]]] = []
        self._filling = False
        self._bulk = False
        self._key_filter: Optional[_TableKeyFilter] = None
        self.last_var = FREQ_VAR
        self.run_update = False
        self.dw.cbSelectType.currentIndexChanged.connect(partial(self.update_traffic_tbl, 'type'))
        self.dw.cbTrafficSelectSeg.currentIndexChanged.connect(partial(self.update_traffic_tbl, 'segment'))
        self.dw.cbTrafficDirectionSelect.currentIndexChanged.connect(partial(self.update_traffic_tbl, 'dir'))
        self._connect_scaling_controls()
        self._wire_table()
        self.set_table_headings()
        self.run_update = True

    def fill_cbTrafficSelectSeg(self):
        """Sets the segment names in cbTrafficSelectSeg"""
        self.dw.cbTrafficSelectSeg.clear()
        nrs = self.dw.twRouteList.rowCount()
        segment_data = getattr(self.omrat, 'segment_data', None)
        for i in range(nrs):
            seg_id = self.dw.twRouteList.item(i, 0).text()
            name_item = self.dw.twRouteList.item(i, 2)
            label = name_item.text() if name_item is not None else seg_id
            if isinstance(segment_data, dict):
                label = f"{label}{status_suffix(seg_id, segment_data)}"
            self.dw.cbTrafficSelectSeg.addItem(label, seg_id)
        self.c_seg = self.dw.cbTrafficSelectSeg.currentData() or ''

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
        self._data_rows = len(types)
        self._data_cols = len(sizes)
        # Table gets one extra row and one extra column for the summary.
        self.dw.twTrafficData.setColumnCount(len(sizes) + 1)
        self.dw.twTrafficData.setHorizontalHeaderLabels(sizes + ['Total'])
        self.dw.twTrafficData.setRowCount(len(types) + 1)
        self.dw.twTrafficData.setVerticalHeaderLabels(types + ['Total'])
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
        tw = self.dw.twTrafficData
        self._filling = True
        try:
            for row in range(len(types)):
                for col in range(len(sizes)):
                    tw.setItem(row, col, self._make_data_item())
                tw.setItem(row, len(sizes), self._make_summary_item())
            for col in range(len(sizes) + 1):
                tw.setItem(len(types), col, self._make_summary_item())
        finally:
            self._filling = False
        self._shown = []
        # Keep the per-type "follow global" checkboxes + the follow_global
        # bool list in sync with the current ship-type count.
        self.populate_scaling_checkboxes()

    def create_empty_dict(self, s_key: str, dirs: list[str]):
        """Creates an empty dict for the segment with all types"""
        self.traffic_data[s_key] = {}
        rows = self._data_rows
        cols = self._data_cols

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

    def update_traffic_tbl(self, caller: str):
        """Updates Traffic data table with the data from traffic_data,
        using the correct type and segment"""
        if self.run_update:
            self.save()
        if caller == 'segment':
            self.c_seg = self.dw.cbTrafficSelectSeg.currentData() or ''
            self.update_direction_select()
        rows = self._data_rows
        cols = self._data_cols
        self.last_var: str = self.dw.cbSelectType.currentText()
        self.c_di = self.dw.cbTrafficDirectionSelect.currentText()
        if any([self.c_seg == "", self.c_di == "", self.last_var == ""]):
            return
        # Make sure the Scaling matrix exists for the (seg, dir) we are
        # about to display.  Legacy projects and freshly-imported IWRAP
        # files may not have it yet.
        self.ensure_scaling_present()
        matrix = self.traffic_data[self.c_seg][self.c_di][self.last_var]
        dp, _gs = self._separators()
        tw = self.dw.twTrafficData
        self._shown = []
        self._filling = True
        try:
            for row in range(rows):
                shown_row: list[tuple[Any, str]] = []
                for col in range(cols):
                    val = matrix[row][col]
                    text = format_number(val, dp)
                    item = tw.item(row, col)
                    if item is None:
                        item = self._make_data_item()
                        tw.setItem(row, col, item)
                    item.setText(text)
                    item.setToolTip(_NO_OBSERVATION_TIP if self._is_unobserved(val) else '')
                    shown_row.append((val, text))
                self._shown.append(shown_row)
        finally:
            self._filling = False
        self._refresh_summary()

    def update_direction_select(self):
        self.run_update = False
        self.dw.cbTrafficDirectionSelect.clear()
        if len(self.traffic_data) == 0 or self.c_seg == '':
            return
        if self.c_seg not in self.traffic_data:
            self.c_seg = ''
            return
        for key in self.traffic_data[self.c_seg].keys():
            self.dw.cbTrafficDirectionSelect.addItem(key)
        self.c_di = self.dw.cbTrafficDirectionSelect.currentText()
        self.run_update = True

    def save(self):
        """Saves the previous "table" in traffic_data"""
        if any([self.c_seg == "", self.c_di == "", self.last_var == ""]):
            return
        seg = self.traffic_data.get(self.c_seg)
        if seg is None:
            return
        di = seg.get(self.c_di)
        if di is None:
            return
        if self.last_var not in di:
            return
        # Edits are written through as they happen (``_commit_cell``); this
        # picks up any cell whose text changed without ``itemChanged``.
        # Untouched cells keep their stored value exactly.
        matrix = di[self.last_var]
        tw = self.dw.twTrafficData
        for row in range(min(self._data_rows, len(self._shown))):
            for col in range(min(self._data_cols, len(self._shown[row]))):
                item = tw.item(row, col)
                if item is None or item.text() == self._shown[row][col][1]:
                    continue
                try:
                    value = self._value_from_text(item.text())
                except ValueError:
                    continue
                matrix[row][col] = value
                self._shown[row][col] = (value, item.text())

    def commit_changes(self):
        """Copy the output from the traffic data within this module to omrats traffic_data"""
        self.save()
        self.omrat.traffic_data = self.traffic_data

    # ------------------------------------------------------- table editing
    # Plain text cells instead of spin boxes (2026-10-16): a block can be
    # selected, copied to Excel, edited there and pasted back.  Numbers
    # are shown with the locale's decimal point (what Excel expects) and
    # parsed with either separator (``table_clipboard``).

    def _wire_table(self) -> None:
        """Connect the table once: cell edits, Ctrl+C / Ctrl+V / Delete and
        the right-click menu.  Mocked widgets in tests just skip."""
        tw = self.dw.twTrafficData
        try:
            # Cell ranges like a spreadsheet (the dock styling makes every
            # other table single-row; ``omrat_widget`` exempts this one).
            tw.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
            tw.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            tw.setCornerButtonEnabled(True)
            tw.itemChanged.connect(self._on_item_changed)
            tw.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            tw.customContextMenuRequested.connect(self._show_table_menu)
            self._key_filter = _TableKeyFilter(self)
            tw.installEventFilter(self._key_filter)
        except (TypeError, AttributeError, RuntimeError):
            pass

    @staticmethod
    def _separators() -> tuple[str, str]:
        """``(decimal point, group separator)`` of the current locale."""
        loc = QLocale()
        return str(loc.decimalPoint()), str(loc.groupSeparator())

    @staticmethod
    def _is_unobserved(val: Any) -> bool:
        if isinstance(val, (list, tuple)):
            return True     # AIS observation list not collapsed yet
        try:
            return not np.isfinite(float(val))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _make_data_item() -> QTableWidgetItem:
        item = QTableWidgetItem('')
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return item

    @staticmethod
    def _make_summary_item(text: str = '-') -> QTableWidgetItem:
        """Non-editable, non-selectable, styled cell of the Total row/column."""
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setBackground(QColor('#dde'))
        item.setForeground(QColor('#333'))
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        return item

    def _current_matrix(self) -> Optional[list]:
        seg = self.traffic_data.get(self.c_seg) if self.c_seg else None
        di = seg.get(self.c_di) if isinstance(seg, dict) else None
        matrix = di.get(self.last_var) if isinstance(di, dict) else None
        return matrix if isinstance(matrix, list) else None

    def _blank_value(self) -> Any:
        """What an emptied cell stores: 0 ships, 100 %, or no observation."""
        if self.last_var == FREQ_VAR:
            return 0
        if self.last_var == SCALING_VAR:
            return SCALING_DEFAULT
        return np.inf

    def _value_from_text(self, text: str) -> Any:
        """Stored value for cell ``text`` in the current variable.  Raises
        ``ValueError`` for text that is not a number >= 0."""
        dp, gs = self._separators()
        v = parse_number(text, dp, gs)
        if v is None:
            return self._blank_value()
        if v < 0:
            raise ValueError(f"negative value: {text!r}")
        if self.last_var == FREQ_VAR and v.is_integer():
            return int(v)
        return v

    def _set_cell_text(self, row: int, col: int, text: str) -> None:
        item = self.dw.twTrafficData.item(row, col)
        if item is None:
            return
        self._filling = True
        try:
            item.setText(text)
        finally:
            self._filling = False

    def _commit_cell(self, row: int, col: int, text: str) -> bool:
        """Parse ``text`` into cell (row, col) of the matrix on screen and
        show it normalised.  Invalid text puts the previous text back and
        returns ``False``."""
        matrix = self._current_matrix()
        if matrix is None or row >= len(self._shown) or col >= len(self._shown[row]):
            return True
        try:
            value = self._value_from_text(text)
        except ValueError:
            self._set_cell_text(row, col, self._shown[row][col][1])
            return False
        matrix[row][col] = value
        shown = format_number(value, self._separators()[0])
        self._set_cell_text(row, col, shown)
        item = self.dw.twTrafficData.item(row, col)
        if item is not None:
            self._filling = True
            try:
                item.setToolTip(_NO_OBSERVATION_TIP if self._is_unobserved(value) else '')
            finally:
                self._filling = False
        self._shown[row][col] = (value, shown)
        if self.last_var == SCALING_VAR:
            self._on_scaling_cell_edited(row, float(value))
        return True

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._filling or item is None:
            return
        row, col = item.row(), item.column()
        if not (0 <= row < min(self._data_rows, len(self._shown)) and 0 <= col < self._data_cols):
            return
        if item.text() == self._shown[row][col][1]:
            return
        typed = item.text()
        if any(ch in typed for ch in '\t\r\n'):
            # A copied block pasted into one cell's editor (its own
            # right-click Paste): paste the clipboard as a block from here.
            self._set_cell_text(row, col, self._shown[row][col][1])
            self.paste_clipboard(anchor=(row, col))
            return
        if not self._commit_cell(row, col, typed):
            self._notify(f"'{typed}' is not a number >= 0 -- the cell keeps its value.", warning=True)
        if not self._bulk:
            self._refresh_summary()

    def _selected_data_cells(self) -> list[tuple[int, int]]:
        """Selected cells inside the data block (totals excluded), sorted."""
        tw = self.dw.twTrafficData
        return sorted({
            (ix.row(), ix.column()) for ix in tw.selectedIndexes()
            if ix.row() < self._data_rows and ix.column() < self._data_cols
        })

    @staticmethod
    def _header_text(item: Optional[QTableWidgetItem]) -> str:
        return item.text() if item is not None else ''

    def copy_selection(self, with_headers: bool = False) -> str:
        """Put the selected cells (or, ``with_headers``, the whole matrix
        with ship-type / length labels) on the clipboard as TSV.  Returns
        the text."""
        tw = self.dw.twTrafficData
        if with_headers:
            rows, cols = range(self._data_rows), range(self._data_cols)
            chosen = None
        else:
            cells = self._selected_data_cells()
            if not cells:
                return ''
            rows = range(min(r for r, _ in cells), max(r for r, _ in cells) + 1)
            cols = range(min(c for _, c in cells), max(c for _, c in cells) + 1)
            chosen = set(cells)
        out: list[list[str]] = []
        if with_headers:
            corner = f"{self.dw.cbTrafficSelectSeg.currentText()} - {self.c_di} - {self.last_var}"
            out.append([corner] + [self._header_text(tw.horizontalHeaderItem(c)) for c in cols])
        for r in rows:
            line = [self._header_text(tw.verticalHeaderItem(r))] if with_headers else []
            for c in cols:
                item = tw.item(r, c)
                keep = item is not None and (chosen is None or (r, c) in chosen)
                line.append(item.text() if keep else '')
            out.append(line)
        text = to_tsv(out)
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        return text

    def paste_clipboard(
        self, text: Optional[str] = None, anchor: Optional[tuple[int, int]] = None,
    ) -> Optional[PastePlan]:
        """Paste TSV (Excel) into the matrix on screen.

        A block with labels (``copy_selection(with_headers=True)``, or a
        labelled range from Excel) lands at the top-left cell; a plain
        block at ``anchor``, else the top-left selected cell; one value
        fills the whole selection.  Cells that are not a number >= 0 keep
        their value.  Always reports in the message bar -- also when
        nothing could be pasted, or nothing changed.
        """
        if text is None:
            clipboard = QApplication.clipboard()
            text = clipboard.text() if clipboard is not None else ''
        if self._current_matrix() is None:
            self._notify("Paste: pick a leg, direction and variable first.", warning=True)
            return None
        dp, gs = self._separators()
        block, had_row, had_col = strip_headers(parse_tsv(text), dp, gs)
        if not block:
            self._notify("Paste: the clipboard holds no table cells.", warning=True)
            return None
        selection = self._selected_data_cells()
        tw = self.dw.twTrafficData
        if anchor is not None:
            top, left = anchor
            selection = []
        elif selection:
            top, left = min(r for r, _ in selection), min(c for _, c in selection)
        else:
            top, left = max(tw.currentRow(), 0), max(tw.currentColumn(), 0)
        start = (0 if had_row else top, 0 if had_col else left)
        plan = plan_paste(block, self._data_rows, self._data_cols, start, selection)
        bad = changed = 0
        self._bulk = True
        try:
            for r, c, cell in plan.cells:
                old = self._shown[r][c][0] if r < len(self._shown) and c < len(self._shown[r]) else None
                if not self._commit_cell(r, c, cell):
                    bad += 1
                elif r < len(self._shown) and c < len(self._shown[r]) and self._differs(old, self._shown[r][c][0]):
                    changed += 1
        finally:
            self._bulk = False
        self._refresh_summary()
        msg = f"Pasted {len(plan.cells) - bad} cell(s) into {self.last_var}, {changed} changed"
        if bad:
            msg += f"; {bad} not a number >= 0 (kept their value)"
        if plan.clipped:
            msg += f"; {plan.clipped} outside the table ignored"
        self._notify(msg, warning=bool(bad))
        return plan

    @staticmethod
    def _differs(old: Any, new: Any) -> bool:
        try:
            return not (old == new or (np.isnan(float(old)) and np.isnan(float(new))))
        except (TypeError, ValueError):
            return True

    def clear_selection(self) -> None:
        """Delete key: empty the selected cells (0 ships, 100 %, or no
        observation for Speed / Draught / heights / beam)."""
        if self._current_matrix() is None:
            return
        self._bulk = True
        try:
            for r, c in self._selected_data_cells():
                self._commit_cell(r, c, '')
        finally:
            self._bulk = False
        self._refresh_summary()

    def _build_table_menu(self) -> QMenu:
        tw = self.dw.twTrafficData
        menu = QMenu(tw)
        for label, slot, key in (
            ("Copy", self.copy_selection, QKeySequence.StandardKey.Copy),
            ("Copy table with headers", partial(self.copy_selection, True), None),
            ("Paste", self.paste_clipboard, QKeySequence.StandardKey.Paste),
            ("Clear", self.clear_selection, QKeySequence.StandardKey.Delete),
            ("Select all", tw.selectAll, QKeySequence.StandardKey.SelectAll),
        ):
            act = menu.addAction(label)
            if key is not None:
                act.setShortcut(QKeySequence(key))
            act.triggered.connect(lambda _checked=False, s=slot: s())
        return menu

    def _show_table_menu(self, pos) -> None:
        tw = self.dw.twTrafficData
        menu = self._build_table_menu()
        menu.exec(tw.viewport().mapToGlobal(pos))
        menu.deleteLater()      # parented to the table: would pile up otherwise

    def _notify(self, message: str, warning: bool = False) -> None:
        notifier = getattr(self.omrat, 'notifier', None)
        if notifier is None:
            return
        try:
            if warning:
                from qgis.core import Qgis
                notifier.display_message(message, level=Qgis.MessageLevel.Warning, duration=8)
            else:
                notifier.display_message(message, duration=5)
        except Exception:  # nosec B110 B112
            pass

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
        rows = self._data_rows
        cols = self._data_cols
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
                n_types = int(self._data_rows)
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
            n_types = int(self._data_rows)
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
            # Re-render so the visible cells pick up the broadcast.
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
        self.ensure_follow_global(self._data_rows)
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

    def _summary_text(self, row: int, col: int, text: str) -> None:
        item = self.dw.twTrafficData.item(row, col)
        if not isinstance(item, QTableWidgetItem):
            return
        self._filling = True
        try:
            item.setText(text)
        finally:
            self._filling = False

    def _refresh_summary(self) -> None:
        """Recompute and display row/column totals in the non-editable summary cells.

        Only meaningful for Frequency (ships/year); all other variables show '-'.
        Called after every table render and after every Frequency edit / paste.
        """
        rows = self._data_rows
        cols = self._data_cols
        if rows == 0 or cols == 0:
            return
        matrix = self._current_matrix() if self.last_var == FREQ_VAR else None
        if matrix is None:
            for row in range(rows):
                self._summary_text(row, cols, '-')
            for col in range(cols + 1):
                self._summary_text(rows, col, '-')
            return
        col_totals = [0.0] * cols
        for row in range(rows):
            row_total = 0.0
            for col in range(cols):
                try:
                    v = float(matrix[row][col])
                except (TypeError, ValueError, IndexError):
                    continue
                if np.isfinite(v):
                    row_total += v
                    col_totals[col] += v
            self._summary_text(row, cols, f'{row_total:.0f}')
        for col in range(cols):
            self._summary_text(rows, col, f'{col_totals[col]:.0f}')
        self._summary_text(rows, cols, f'{sum(col_totals):.0f}')

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
        tw = self.dw.twTrafficData
        for sig, slot in ((tw.itemChanged, self._on_item_changed),
                          (tw.customContextMenuRequested, self._show_table_menu)):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError, AttributeError):
                pass
        if self._key_filter is not None:
            try:
                tw.removeEventFilter(self._key_filter)
            except (TypeError, RuntimeError, AttributeError):
                pass
            self._key_filter = None
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
