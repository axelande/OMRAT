"""Tests for IWRAP XML import functionality.

This module tests the import of IWRAP XML files into .omrat format,
with special attention to handling missing fields like dist1 and dist2.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Mock pytest.fail for direct execution
    class pytest:
        @staticmethod
        def fail(msg):
            raise AssertionError(msg)

from compute.iwrap_convertion import parse_iwrap_xml, read_iwrap_xml


def test_parse_case2_23_xml():
    """Test parsing the case2_23.xml file without crashing on missing fields."""
    xml_path = 'tests/example_data/case2_23.xml'

    # Verify file exists
    assert os.path.exists(xml_path), f"Test file {xml_path} not found"

    # Parse the XML file
    result = parse_iwrap_xml(xml_path)

    # Verify basic structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert 'project_name' in result, "Result should have project_name"
    assert 'traffic_data' in result, "Result should have traffic_data"
    assert 'segment_data' in result, "Result should have segment_data"
    assert 'drift' in result, "Result should have drift"
    assert 'depths' in result, "Result should have depths"
    assert 'objects' in result, "Result should have objects"

    # Verify project name
    assert result['project_name'] == 'case2_23', f"Expected project name 'case2_23', got '{result['project_name']}'"

    print(f"✓ Successfully parsed {xml_path}")
    print(f"  - Project: {result['project_name']}")
    print(f"  - Segments: {len(result['segment_data'])}")
    print(f"  - Depth areas: {len(result['depths'])}")
    print(f"  - Objects: {len(result['objects'])}")


def test_missing_dist_fields_handled():
    """Test that missing dist1 and dist2 fields are added as empty lists."""
    xml_path = 'tests/example_data/case2_23.xml'

    result = parse_iwrap_xml(xml_path)

    # Check segments for dist1/dist2 fields
    for seg_id, segment in result['segment_data'].items():
        # dist1 and dist2 should be present as empty lists (required by schema)
        assert 'dist1' in segment, f"Segment {seg_id} should have dist1 field"
        assert 'dist2' in segment, f"Segment {seg_id} should have dist2 field"
        assert isinstance(segment['dist1'], list), f"dist1 should be a list"
        assert isinstance(segment['dist2'], list), f"dist2 should be a list"

        # Verify expected fields are present
        assert 'Leg_name' in segment, f"Segment {seg_id} should have Leg_name"
        assert 'Start_Point' in segment, f"Segment {seg_id} should have Start_Point"
        assert 'End_Point' in segment, f"Segment {seg_id} should have End_Point"
        assert 'Dirs' in segment, f"Segment {seg_id} should have Dirs field"

    print(f"✓ dist1/dist2 fields added as empty lists (schema requirement)")


def test_drift_settings_imported():
    """Test that drifting settings are imported correctly."""
    xml_path = 'tests/example_data/case2_23.xml'

    result = parse_iwrap_xml(xml_path)

    # Verify drift data exists
    assert 'drift' in result, "Result should have drift settings"
    drift = result['drift']

    # Check for common drift fields
    if 'speed' in drift:
        assert isinstance(drift['speed'], (int, float)), "Drift speed should be numeric"
        print(f"  - Drift speed: {drift['speed']}")

    if 'anchor_d' in drift:
        assert isinstance(drift['anchor_d'], (int, float)), "Anchor depth should be numeric"
        print(f"  - Anchor depth: {drift['anchor_d']}")

    if 'rose' in drift:
        assert isinstance(drift['rose'], dict), "Rose should be a dictionary"
        print(f"  - Wind rose directions: {len(drift['rose'])}")

    print(f"✓ Drift settings imported successfully")


def test_segment_data_imported():
    """Test that segment data (legs) are imported correctly."""
    xml_path = 'tests/example_data/case2_23.xml'

    result = parse_iwrap_xml(xml_path)

    segments = result['segment_data']
    assert len(segments) > 0, "Should have at least one segment"

    # Check first segment structure
    first_seg = next(iter(segments.values()))

    # Required fields
    required_fields = ['Leg_name', 'Start_Point', 'End_Point']
    for field in required_fields:
        assert field in first_seg, f"Segment should have {field}"

    # Optional fields that might be present
    if 'Width' in first_seg:
        assert isinstance(first_seg['Width'], (int, float)), "Width should be numeric"

    # Check for distribution parameters (if present)
    for i in range(1, 4):
        if f'mean1_{i}' in first_seg:
            assert isinstance(first_seg[f'mean1_{i}'], (int, float)), f"mean1_{i} should be numeric"
        if f'std1_{i}' in first_seg:
            assert isinstance(first_seg[f'std1_{i}'], (int, float)), f"std1_{i} should be numeric"
        if f'weight1_{i}' in first_seg:
            assert isinstance(first_seg[f'weight1_{i}'], (int, float)), f"weight1_{i} should be numeric"

    print(f"✓ Segment data imported successfully ({len(segments)} segments)")


def test_areas_imported():
    """Test that depth areas and objects are imported correctly."""
    xml_path = 'tests/example_data/case2_23.xml'

    result = parse_iwrap_xml(xml_path)

    # Check depths - should be list of lists: [id, depth, polygon]
    depths = result['depths']
    for depth in depths:
        assert isinstance(depth, list), "Depth should be a list"
        assert len(depth) == 3, "Depth should have 3 elements: [id, depth, polygon]"
        depth_id, depth_val, polygon = depth

        # Verify polygon format
        if polygon:
            assert polygon.startswith('POLYGON'), "Polygon should be in WKT format"

    # Check objects - should be list of lists: [id, height, polygon]
    objects = result['objects']
    for obj in objects:
        assert isinstance(obj, list), "Object should be a list"
        assert len(obj) == 3, "Object should have 3 elements: [id, height, polygon]"

    print(f"✓ Areas imported successfully ({len(depths)} depths, {len(objects)} objects)")


def test_write_to_file():
    """Test writing imported data to a .omrat file."""
    xml_path = 'tests/example_data/case2_23.xml'
    output_path = 'tests/example_data/imported_case2_23.omrat'

    try:
        # Import and write to file
        read_iwrap_xml(xml_path, output_path)

        # Verify file was created
        assert os.path.exists(output_path), f"Output file {output_path} was not created"

        # Verify file is valid JSON
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert isinstance(data, dict), "Output file should contain a dictionary"
        assert 'project_name' in data, "Output should have project_name"

        print(f"✓ Successfully wrote imported data to {output_path}")

    finally:
        # Clean up test file
        if os.path.exists(output_path):
            os.remove(output_path)


def test_round_trip_export_import():
    """Test exporting and then importing produces consistent data."""
    from compute.iwrap_convertion import write_iwrap_xml

    # Use existing proj.omrat file
    original_path = 'tests/example_data/proj.omrat'
    export_path = 'tests/example_data/test_export.xml'
    import_path = 'tests/example_data/test_import.omrat'

    try:
        # Export to XML
        write_iwrap_xml(original_path, export_path)
        assert os.path.exists(export_path), "Export file was not created"

        # Import back to .omrat
        read_iwrap_xml(export_path, import_path)
        assert os.path.exists(import_path), "Import file was not created"

        # Load both files
        with open(original_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        with open(import_path, 'r', encoding='utf-8') as f:
            imported_data = json.load(f)

        # Verify key structures exist
        assert 'project_name' in imported_data
        assert 'segment_data' in imported_data
        assert 'drift' in imported_data

        # Check segment count matches
        assert len(imported_data['segment_data']) == len(original_data.get('segment_data', {})), \
            "Segment count should match after round-trip"

        print(f"✓ Round-trip export/import successful")
        print(f"  - Original segments: {len(original_data.get('segment_data', {}))}")
        print(f"  - Imported segments: {len(imported_data['segment_data'])}")

    finally:
        # Clean up test files
        for path in [export_path, import_path]:
            if os.path.exists(path):
                os.remove(path)


def test_no_crash_on_empty_elements():
    """Test that import handles empty or missing XML elements gracefully."""
    xml_path = 'tests/example_data/case2_23.xml'

    # This should not raise any exceptions
    try:
        result = parse_iwrap_xml(xml_path)
        assert isinstance(result, dict), "Should return a dictionary even with missing elements"
        print(f"✓ No crashes on empty elements")
    except Exception as e:
        pytest.fail(f"Import crashed on empty elements: {e}")


if __name__ == '__main__':
    print("Running IWRAP import tests...\n")

    # Run all tests
    test_parse_case2_23_xml()
    test_missing_dist_fields_handled()
    test_drift_settings_imported()
    test_segment_data_imported()
    test_areas_imported()
    test_write_to_file()
    test_round_trip_export_import()
    test_no_crash_on_empty_elements()

    print("\n✅ All tests passed!")
