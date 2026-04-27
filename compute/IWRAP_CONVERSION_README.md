# IWRAP XML Import/Export

This module provides bidirectional conversion between OMRAT's `.omrat` JSON format and IWRAP's XML format.

## Features

- **Export**: Convert `.omrat` files to IWRAP XML format
- **Import**: Convert IWRAP XML files to `.omrat` format
- **Error Handling**: Gracefully handles missing fields like `dist1` and `dist2` that don't exist in IWRAP XML files
- **Round-trip Support**: Export and re-import with data integrity

## Usage

### Command Line Interface

#### Export .omrat to IWRAP XML

```bash
python compute/iwrap_convertion.py export input.omrat output.xml
```

#### Import IWRAP XML to .omrat

```bash
python compute/iwrap_convertion.py import input.xml output.omrat
```

### Python API

#### Export

```python
from compute.iwrap_convertion import write_iwrap_xml

write_iwrap_xml('project.omrat', 'output.xml')
```

#### Import

```python
from compute.iwrap_convertion import read_iwrap_xml

read_iwrap_xml('input.xml', 'project.omrat')
```

#### Parse XML to Dictionary

```python
from compute.iwrap_convertion import parse_iwrap_xml

data = parse_iwrap_xml('input.xml')
# Returns a dictionary in .omrat format
print(data['project_name'])
print(len(data['segment_data']))
```

## Data Mapping

### Export (.omrat → IWRAP XML)

| .omrat Field | IWRAP XML Element | Notes |
|--------------|-------------------|-------|
| `project_name` | `<riskmodel name="...">` | Root element |
| `segment_data` | `<legs>` | Route segments |
| `drift` | `<drifting>` | Drift settings and wind rose |
| `depths` | `<areas>` with `type="0"` | Depth areas |
| `objects` | `<areas>` with `type="1"` | Objects as areas with depth=-1 |
| `traffic_data` | `<traffic_distributions>` | Ship traffic data |

### Import (IWRAP XML → .omrat)

| IWRAP XML Element | .omrat Field | Notes |
|-------------------|--------------|-------|
| `<riskmodel name="...">` | `project_name` | Root element |
| `<legs>` | `segment_data` | Route segments |
| `<waypoints>` | Used in segment endpoints | Converted to Start_Point/End_Point |
| `<manoeuvring_aspects_legs>` | Distribution parameters in segments | mean1_i, std1_i, weight1_i, etc. |
| `<drifting>` | `drift` | Drift settings |
| `<areas>` | `depths` or `objects` | Separated by depth value |
| `<traffic_distributions>` | Not fully implemented | Categories extracted |

## Important Notes

### Missing Fields

The IWRAP XML format does not contain certain fields present in `.omrat` files:

- **`dist1` and `dist2`**: These fields are OMRAT-specific and are not present in IWRAP XML. The import function handles their absence gracefully.
- **Object heights**: IWRAP stores objects as areas with `depth=-1` but doesn't preserve height information.

### Known Limitations

1. **Traffic data**: The import currently extracts category data but may not fully reconstruct the original traffic matrix structure.
2. **Ship categories**: The mapping from IWRAP ship types to OMRAT categories is approximate.
3. **Bridges**: Not currently implemented in import (rarely used).

## Testing

Run the comprehensive test suite:

```bash
python3 tests/test_iwrap_import.py
```

Tests include:
- ✓ Parsing case2_23.xml without crashes
- ✓ Handling missing dist1/dist2 fields
- ✓ Importing drift settings
- ✓ Importing segment data
- ✓ Importing depth areas and objects
- ✓ Writing to .omrat files
- ✓ Round-trip export/import consistency

## Example Workflow

### Scenario: Import IWRAP project and modify in OMRAT

```bash
# 1. Import IWRAP XML to OMRAT format
python compute/iwrap_convertion.py import original.xml project.omrat

# 2. Modify project.omrat using OMRAT tools
# ... (your modifications) ...

# 3. Export back to IWRAP XML
python compute/iwrap_convertion.py export project.omrat modified.xml
```

### Scenario: Batch conversion

```python
import os
from compute.iwrap_convertion import read_iwrap_xml

# Convert all XML files in a directory
xml_dir = 'iwrap_projects'
output_dir = 'omrat_projects'

for filename in os.listdir(xml_dir):
    if filename.endswith('.xml'):
        xml_path = os.path.join(xml_dir, filename)
        omrat_path = os.path.join(output_dir, filename.replace('.xml', '.omrat'))
        read_iwrap_xml(xml_path, omrat_path)
        print(f"✓ Converted {filename}")
```

## Troubleshooting

### Import fails with XML parsing error

Ensure the XML file is valid IWRAP format. Check for:
- Proper XML encoding (UTF-8)
- Well-formed XML structure
- Valid IWRAP schema elements

### Missing data after import

Some IWRAP elements may not have equivalents in OMRAT format. Check the console output for warnings.

### Round-trip data loss

Certain OMRAT-specific fields (like `dist1`, `dist2`) cannot be stored in IWRAP XML and will be lost on export.

## Support

For issues or questions, please report them in the project issue tracker.
