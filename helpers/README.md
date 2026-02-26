# OMRAT Helpers Module

This directory contains helper utilities for the OMRAT plugin.

## Modules

### qt_conversions.py

Qt5/Qt6 compatibility layer that handles API differences between Qt versions through QGIS's PyQt abstraction.

#### Quick Start

```python
from helpers.qt_conversions import QtCompat, is_qt6, is_qt5, get_qt_version

# Check Qt version
print(f"Running Qt {get_qt_version()}")
if is_qt6():
    print("Qt6 detected")
else:
    print("Qt5 detected")
```

#### QtCompat Class Methods

**Version Detection:**
- `QT_VERSION` - String: Qt version (e.g., "5.15.13")
- `QT_MAJOR_VERSION` - Int: Major version number
- `IS_QT6` - Bool: True if Qt6, False if Qt5

**Enum Handling:**
```python
# Convert Qt enums to integers (Qt5/Qt6 compatible)
value = QtCompat.to_int(QMetaType.QString)
```

**Dialog Execution:**
```python
# Qt5 uses exec_(), Qt6 uses exec()
result = QtCompat.exec_dialog(my_dialog)
```

**QVariant Types:**
```python
# Get QVariant type for Qt5/Qt6
string_type = QtCompat.get_variant_type('String')
int_type = QtCompat.get_variant_type('Int')
double_type = QtCompat.get_variant_type('Double')
```

**File Dialogs:**
```python
# Compatible file dialogs
filename, filter = QtCompat.get_save_file_name(
    parent=self,
    caption='Save File',
    directory='/home',
    filter='All Files (*.*)'
)

filename, filter = QtCompat.get_open_file_name(
    parent=self,
    caption='Open File',
    directory='/home',
    filter='All Files (*.*)'
)
```

**String Conversion:**
```python
# QString to Python string (Qt5/Qt6 compatible)
python_str = QtCompat.from_qstring(qstring_obj)
```

**QByteArray Creation:**
```python
# Create QByteArray from bytes or string
qba = QtCompat.to_qbytearray(b'data')
qba = QtCompat.to_qbytearray('string')
```

## Usage Examples

### Basic Import Pattern

Always use QGIS's PyQt abstraction instead of direct PyQt5/PyQt6 imports:

```python
# ✅ Good - Qt5/Qt6 compatible
from qgis.PyQt.QtWidgets import QLabel, QMessageBox
from qgis.PyQt.QtCore import QSettings, Qt
from qgis.PyQt.QtGui import QIcon

# ❌ Bad - Qt5 only
from PyQt5.QtWidgets import QLabel, QMessageBox
from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtGui import QIcon

# ❌ Bad - Qt6 only
from PyQt6.QtWidgets import QLabel, QMessageBox
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QIcon
```

### Handling Version-Specific Code

When you need to handle Qt5/Qt6 differences:

```python
from qgis.PyQt.QtCore import QMetaType
from helpers.qt_conversions import QtCompat, is_qt6

# Example: Adding a field to a QGIS layer
from qgis.core import QgsField

if is_qt6():
    # Qt6 approach
    field = QgsField("name", QMetaType.Type.QString)
else:
    # Qt5 approach
    from qgis.PyQt.QtCore import QVariant
    field = QgsField("name", QVariant.String)

# Or use QtCompat helper:
field_type = QtCompat.get_variant_type('String')
field = QgsField("name", field_type)
```

### Dialog Execution

```python
from qgis.PyQt.QtWidgets import QDialog
from helpers.qt_conversions import QtCompat

class MyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        # dialog setup...

# Show dialog in a Qt5/Qt6 compatible way
dialog = MyDialog()
result = QtCompat.exec_dialog(dialog)
if result == QDialog.Accepted:
    # User clicked OK
    pass
```

### Enum Conversions

```python
from qgis.PyQt.QtCore import QMetaType
from helpers.qt_conversions import QtCompat

# Qt6 uses proper enums that need .value
# Qt5 uses integer constants
# QtCompat handles both:

meta_type = QMetaType.QString
type_id = QtCompat.to_int(meta_type)
```

## When to Use QtCompat

Use `QtCompat` when you encounter:

1. **Enum conversions** - Qt6 enums need `.value`, Qt5 doesn't
2. **Dialog execution** - `exec_()` vs `exec()`
3. **QVariant types** - Different APIs between Qt5/Qt6
4. **QString handling** - Qt5 has QString, Qt6 doesn't
5. **Any API that changed between Qt5 and Qt6**

For standard widgets and signals/slots, `qgis.PyQt` imports are sufficient.

## Testing

Test your code works with both Qt versions:

```python
from helpers.qt_conversions import get_qt_version, is_qt6

def test_qt_compatibility():
    print(f"Testing with Qt {get_qt_version()}")
    if is_qt6():
        print("Running Qt6-specific tests")
    else:
        print("Running Qt5-specific tests")
```

## Contributing

When adding new Qt compatibility code:

1. Add the method to `QtCompat` class in `qt_conversions.py`
2. Document the method with docstring
3. Export it in `__init__.py` if it's a top-level function
4. Update this README with usage examples

## References

- [QGIS PyQt API](https://qgis.org/pyqgis/)
- [Qt5 to Qt6 Porting Guide](https://doc.qt.io/qt-6/portingguide.html)
- Main documentation: [MIGRATION_NOTES.md](../MIGRATION_NOTES.md)
- Summary: [PYQT_MIGRATION_SUMMARY.md](../PYQT_MIGRATION_SUMMARY.md)
