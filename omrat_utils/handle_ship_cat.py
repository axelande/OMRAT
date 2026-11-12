from typing import Any, Optional

from qgis.PyQt.QtWidgets import QAbstractButton, QTableWidget, QTableWidgetItem

from ui.ship_categories_widget import ShipCategoriesWidget


class ShipCategories:
    """Settings > Ship Categories.

    The dialog's tables (``cvTypes``, ``twLengths``) *are* the stored ship
    categories: ``GatherData.get_ship_categories_for_save`` and the
    consequence handler read them directly, so an edit counts as soon as it
    is typed.  To make Cancel mean "no change", ``run`` copies the tables
    and buttons when the dialog opens and ``discard_changes`` puts the copy
    back when the dialog is rejected (Cancel, Esc or the window's close
    button); OK keeps the edits.
    """

    def __init__(self, parent):
        self.parent = parent
        self.scw = ShipCategoriesWidget(None)
        self._before: Optional[dict[str, Any]] = None

    def commit_changes(self) -> None:
        """OK: the edits are already in the tables -- forget the copy."""
        self._before = None

    def discard_changes(self) -> None:
        """Cancel / Esc / close: restore the tables as they were when opened."""
        if self._before is not None:
            self._restore(self._before)
        self._before = None

    def run(self):
        self._before = self._snapshot()
        self.scw.show()
        self._wire_signals()
        self.scw.exec()

    def _wire_signals(self) -> None:
        """Connect OK / Cancel -- once (``run`` is called on every open of the
        same dialog; connecting there added one connection per open).

        The dialog's own ``accepted`` / ``rejected`` (not the button box's):
        Esc and the window's close button reject the dialog without going
        through the Cancel button.
        """
        if getattr(self, '_signals_wired', False):
            return
        self._signals_wired = True
        self.scw.accepted.connect(self.commit_changes)
        self.scw.rejected.connect(self.discard_changes)

    # ------------------------------------------------------------------
    # Copy / restore of everything the user can change in the dialog
    # ------------------------------------------------------------------
    def _snapshot(self) -> dict[str, Any]:
        tables = {}
        for tw in self.scw.findChildren(QTableWidget):
            tables[tw.objectName()] = (tw.columnCount(), [
                [tw.item(r, c).text() if tw.item(r, c) is not None else None
                 for c in range(tw.columnCount())]
                for r in range(tw.rowCount())
            ])
        checked = {b.objectName(): b.isChecked()
                   for b in self.scw.findChildren(QAbstractButton) if b.isCheckable()}
        return {'tables': tables, 'checked': checked}

    def _restore(self, snap: dict[str, Any]) -> None:
        for tw in self.scw.findChildren(QTableWidget):
            if tw.objectName() not in snap['tables']:
                continue
            n_cols, rows = snap['tables'][tw.objectName()]
            tw.setColumnCount(n_cols)
            tw.setRowCount(len(rows))
            for r, row in enumerate(rows):
                for c, text in enumerate(row):
                    item = tw.item(r, c)
                    if text is None:
                        tw.takeItem(r, c)
                    elif item is None:
                        tw.setItem(r, c, QTableWidgetItem(text))
                    elif item.text() != text:
                        item.setText(text)
        for b in self.scw.findChildren(QAbstractButton):
            if b.isCheckable() and b.objectName() in snap['checked']:
                b.setChecked(snap['checked'][b.objectName()])
