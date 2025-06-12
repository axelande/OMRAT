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
        self.dsw.show()
        # Get the button box
        self.buttonBox = self.dsw.findChild(QDialogButtonBox, 'buttonBox')
        
        # Connect the accepted signal to your custom slot
        self.buttonBox.ok.connect(self.commit_changes)
        
        # Optionally, connect the rejected signal to a different slot
        self.buttonBox.cancel.connect(self.discard_changes)
        self.dsw.exec_()