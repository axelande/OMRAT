# Testing the IWRAP Import Menu Feature

## What Was Added

A new menu option **"Import from IWRAP XML"** has been added to the File menu in OMRAT, parallel to the existing "Export to IWRAP XML" option.

## File Menu Structure

```
File
├── Save
├── Load
├── ─────────────
├── Export to IWRAP XML
└── Import from IWRAP XML  ← NEW!
```

## How It Works

### User Workflow

1. **Open OMRAT** in QGIS
2. **Click File → Import from IWRAP XML**
3. **Select an IWRAP XML file** (e.g., `case2_23.xml`)
4. **System will:**
   - Parse the XML file
   - Convert it to .omrat format
   - Validate the data
   - Load it into the current session
   - Show a success message with project details

### What Gets Imported

The import function converts IWRAP XML data to OMRAT format, including:

- ✓ Project metadata (name, version)
- ✓ Route segments (legs) with waypoints
- ✓ Traffic distributions and ship categories
- ✓ Manoeuvring aspects and distribution parameters
- ✓ Drift settings (speed, wind rose, repair time)
- ✓ Depth areas
- ✓ Objects (imported as areas with depth=-1)
- ✓ Causation factors

### Error Handling

The import function gracefully handles:

- **Missing fields**: Fields like `dist1` and `dist2` that don't exist in IWRAP XML files
- **Empty elements**: Optional XML elements that may be missing
- **Validation errors**: Shows a warning but still loads the data
- **File not found**: Shows an error message if the file doesn't exist

## Code Changes Made

### 1. Added Import to `omrat.py`

**Line 47**: Added import statement
```python
from compute.iwrap_convertion import write_iwrap_xml, parse_iwrap_xml
```

**Line 816**: Added menu action
```python
fileMenu.addAction("Import from IWRAP XML", self.import_from_iwrap)
```

**Lines 561-622**: Added import method
```python
def import_from_iwrap(self) -> None:
    """Import project data from IWRAP XML format."""
    # Opens file dialog, parses XML, validates, and loads data
```

### 2. Import Function Features

The `import_from_iwrap` method:
- Opens a file dialog filtered for XML files
- Parses IWRAP XML using `parse_iwrap_xml()`
- Normalizes and validates the data
- Populates the plugin using `GatherData.populate()`
- Shows success/error messages to the user

## Testing Instructions

### Test with Sample File

1. **Start QGIS** with the OMRAT plugin loaded
2. **Navigate to** File → Import from IWRAP XML
3. **Select** `tests/example_data/case2_23.xml`
4. **Verify** success message shows:
   - Project name: "case2_23"
   - Number of segments: 8
5. **Check** that data appears in the OMRAT interface:
   - Route segments loaded
   - Traffic data present
   - Drift settings imported

### Test with Your Own IWRAP File

1. Export a project from IWRAP as XML
2. Import it into OMRAT using File → Import from IWRAP XML
3. Verify all expected data is present
4. (Optional) Export back to XML to verify round-trip consistency

## Known Limitations

See [IWRAP_CONVERSION_README.md](../compute/IWRAP_CONVERSION_README.md) for details on:
- Fields that don't exist in IWRAP XML (dist1, dist2)
- Data that may not be fully preserved (object heights)
- Traffic data reconstruction limitations

## Troubleshooting

### Import button doesn't appear
- Ensure you're using the latest version of the plugin
- Restart QGIS to reload the plugin

### Import fails with error
- Check the QGIS Python Console for detailed error messages
- Verify the XML file is valid IWRAP format
- Try exporting a simple project from IWRAP first

### Data missing after import
- Some IWRAP elements may not have direct equivalents
- Check the validation warnings in the success dialog
- Review the imported data in each tab

## Related Files

- [iwrap_convertion.py](../compute/iwrap_convertion.py) - Import/export functions
- [test_iwrap_import.py](test_iwrap_import.py) - Test suite
- [IWRAP_CONVERSION_README.md](../compute/IWRAP_CONVERSION_README.md) - Full documentation

## Command Line Alternative

You can also import IWRAP XML files from the command line:

```bash
python compute/iwrap_convertion.py import input.xml output.omrat
```

Then open the resulting `.omrat` file using File → Load in OMRAT.
