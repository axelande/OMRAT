from qgis.PyQt.QtWidgets import QDialogButtonBox

from ui.ship_categories_widget import ShipCategoriesWidget

class ShipCategories:
    def __init__(self, parent):
        self.parent = parent
        self.scw = ShipCategoriesWidget(None)
        
    def commit_changes(self):
        pass
        
    def discard_changes(self):
        pass
        
    def run(self):
        self.scw.show()
        self.buttonBox = self.scw.findChild(QDialogButtonBox, 'buttonBox')
        if self.buttonBox is not None:
            # ``QDialogButtonBox`` exposes ``accepted`` / ``rejected`` --
            # there are no ``ok`` / ``cancel`` signals.
            self.buttonBox.accepted.connect(self.commit_changes)
            self.buttonBox.rejected.connect(self.discard_changes)
        # ``exec_`` exists in PyQt5 but was dropped in PyQt6 (QGIS 4).
        # ``exec`` is present in both, so prefer that.
        self.scw.exec()