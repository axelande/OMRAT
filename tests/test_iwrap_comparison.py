# -*- coding: utf-8 -*-
"""
Test script to compare OMRAT calculation results with IWRAP.

IWRAP reference values:
- Grounding probability: 1.62e-07
- Allision probability: 0.0351

Target: Results within ±50% of IWRAP values.
"""

import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch


def load_omrat_data(filepath: str) -> dict:
    """Load .omrat project file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_data_structure(data: dict) -> dict:
    """Analyze the data structure and extract key metrics."""
    info = {}

    # Depth contours
    depths = data.get('depths', [])
    depth_vals = sorted(set(float(d[1]) for d in depths))
    info['depth_contours'] = depth_vals
    info['num_depths'] = len(depths)

    # Structures
    objects = data.get('objects', [])
    info['num_structures'] = len(objects)
    info['structure_heights'] = [float(o[1]) for o in objects]

    # Traffic data summary
    traffic_data = data.get('traffic_data', {})
    info['num_cells'] = len(traffic_data)

    # Collect all draughts and heights
    all_draughts = set()
    all_heights = set()
    total_ships = 0

    for cell_key, cell_data in traffic_data.items():
        if isinstance(cell_data, dict):
            for cat_key, cat_data in cell_data.items():
                if isinstance(cat_data, dict):
                    draughts = cat_data.get('Draught (meters)', [])
                    heights = cat_data.get('Ship heights (meters)', [])
                    freqs = cat_data.get('Frequency (ships/year)', [])

                    for row_idx, row in enumerate(draughts):
                        if isinstance(row, list):
                            for col_idx, val in enumerate(row):
                                if val != float('inf') and val > 0:
                                    all_draughts.add(val)
                                    # Count ships
                                    if row_idx < len(freqs) and col_idx < len(freqs[row_idx]):
                                        freq = freqs[row_idx][col_idx]
                                        if isinstance(freq, (int, float)) and freq > 0:
                                            total_ships += freq

                    for row in heights:
                        if isinstance(row, list):
                            for val in row:
                                if val != float('inf') and val > 0:
                                    all_heights.add(val)

    info['unique_draughts'] = sorted(all_draughts)
    info['unique_heights'] = sorted(all_heights)
    info['num_unique_draughts'] = len(all_draughts)
    info['num_unique_heights'] = len(all_heights)
    info['total_ships_per_year'] = total_ships

    # Drift parameters
    drift = data.get('drift', {})
    info['drift_prob'] = drift.get('drift_p', 0)
    info['anchor_prob'] = drift.get('anchor_p', 0)
    info['anchor_depth'] = drift.get('anchor_d', 0)
    info['drift_speed'] = drift.get('drift_v', 0)

    return info


def create_smart_bins(draughts: list[float], depth_contours: list[float]) -> dict[float, list[float]]:
    """
    Create smart bins mapping depth thresholds to lists of draughts.

    Ships only ground where their draught > water depth.
    So we only need to calculate separately for each unique depth contour.

    Args:
        draughts: List of unique ship draughts
        depth_contours: List of depth contour values

    Returns:
        Dict mapping depth threshold to list of draughts that would be affected
    """
    bins = {}

    for depth in depth_contours:
        # Ships with draught > depth would ground at this depth
        affected_draughts = [d for d in draughts if d > depth]
        if affected_draughts:
            bins[depth] = affected_draughts

    return bins


def create_height_bins(heights: list[float], structure_heights: list[float]) -> dict[float, list[float]]:
    """
    Create smart bins for structure heights.

    Ships only collide with structures taller than their air draught.

    Args:
        heights: List of unique ship heights
        structure_heights: List of structure height values

    Returns:
        Dict mapping structure height threshold to list of ship heights affected
    """
    bins = {}

    for struct_h in structure_heights:
        # Ships with height < structure height could collide
        affected_heights = [h for h in heights if h < struct_h]
        if affected_heights:
            bins[struct_h] = affected_heights

    return bins


class TestDataAnalysis:
    """Test data analysis functions."""

    @pytest.fixture
    def project_data(self):
        """Load the example project data."""
        data_path = Path(__file__).parent / 'example_data' / 'proj.omrat'
        return load_omrat_data(str(data_path))

    def test_load_project(self, project_data):
        """Verify project loads correctly."""
        assert 'traffic_data' in project_data
        assert 'depths' in project_data
        assert 'objects' in project_data

    def test_analyze_structure(self, project_data):
        """Analyze and print data structure."""
        info = analyze_data_structure(project_data)

        print("\n=== Data Structure Analysis ===")
        print(f"Depth contours: {info['depth_contours']}")
        print(f"Number of depth entries: {info['num_depths']}")
        print(f"Number of structures: {info['num_structures']}")
        print(f"Structure heights: {info['structure_heights']}")
        print(f"Number of traffic cells: {info['num_cells']}")
        print(f"Unique draughts: {info['num_unique_draughts']}")
        print(f"Unique heights: {info['num_unique_heights']}")
        print(f"Total ships/year: {info['total_ships_per_year']}")
        print(f"Drift probability: {info['drift_prob']}")
        print(f"Anchor probability: {info['anchor_prob']}")
        print(f"Anchor depth: {info['anchor_depth']}")
        print(f"Drift speed: {info['drift_speed']}")

        # Verify expected structure
        assert info['num_depths'] == 17
        assert info['num_structures'] == 2
        assert all(h == 10.0 for h in info['structure_heights'])

    def test_smart_binning(self, project_data):
        """Test smart binning for draught/depth calculations."""
        info = analyze_data_structure(project_data)

        depth_bins = create_smart_bins(
            info['unique_draughts'],
            info['depth_contours']
        )

        print("\n=== Smart Draught Binning ===")
        for depth, draughts in sorted(depth_bins.items()):
            print(f"Depth {depth}m: {len(draughts)} ships affected")

        # With depth contours at 0,3,6,9,12...
        # Ships with draught > 0 would be affected by 0m depth
        assert 0.0 in depth_bins
        assert len(depth_bins[0.0]) == len(info['unique_draughts'])

        # Ships with draught > 15m would be fewer
        if 15.0 in depth_bins:
            assert len(depth_bins[15.0]) < len(depth_bins[0.0])

    def test_height_binning(self, project_data):
        """Test smart binning for height/structure calculations."""
        info = analyze_data_structure(project_data)

        height_bins = create_height_bins(
            info['unique_heights'],
            info['structure_heights']
        )

        print("\n=== Smart Height Binning ===")
        for struct_h, heights in sorted(height_bins.items()):
            print(f"Structure {struct_h}m: {len(heights)} ships affected")

        # Structure height is 10m, so ships with height < 10m would be affected
        assert 10.0 in height_bins


class TestCalculation:
    """Test the actual calculation against IWRAP values."""

    # IWRAP reference values
    IWRAP_GROUNDING = 1.62e-07
    IWRAP_ALLISION = 0.0351
    TOLERANCE = 0.5  # 50%

    @pytest.fixture
    def project_data(self):
        """Load the example project data."""
        data_path = Path(__file__).parent / 'example_data' / 'proj.omrat'
        return load_omrat_data(str(data_path))

    @pytest.fixture
    def mock_omrat(self, project_data):
        """Create a mock OMRAT instance for calculation."""
        # Import here to avoid import issues
        from compute.run_calculations import Calculation

        # Create mock parent
        mock_parent = MagicMock()
        mock_parent.main_widget = MagicMock()
        mock_parent.main_widget.LEPDriftAllision = MagicMock()
        mock_parent.main_widget.LEPDriftAllision.setText = MagicMock()
        mock_parent.main_widget.LEPDriftingGrounding = MagicMock()
        mock_parent.main_widget.LEPDriftingGrounding.setText = MagicMock()
        mock_parent.main_widget.cbShipCategories = MagicMock()
        mock_parent.main_widget.cbShipCategories.count = MagicMock(return_value=0)
        mock_parent.main_widget.LEReportPath = MagicMock()
        mock_parent.main_widget.LEReportPath.text = MagicMock(return_value='')

        calc = Calculation(mock_parent)
        return calc, project_data

    def test_run_calculation(self, mock_omrat):
        """Run the calculation and compare with IWRAP values."""
        calc, data = mock_omrat

        print("\n=== Running OMRAT Calculation ===")

        # Run the calculation
        allision, grounding = calc.run_drifting_model(data)

        print(f"\nOMRAT Results:")
        print(f"  Allision probability: {allision:.3e}")
        print(f"  Grounding probability: {grounding:.3e}")
        print(f"\nIWRAP Reference:")
        print(f"  Allision probability: {self.IWRAP_ALLISION:.3e}")
        print(f"  Grounding probability: {self.IWRAP_GROUNDING:.3e}")

        # Calculate ratios
        allision_ratio = allision / self.IWRAP_ALLISION if self.IWRAP_ALLISION > 0 else float('inf')
        grounding_ratio = grounding / self.IWRAP_GROUNDING if self.IWRAP_GROUNDING > 0 else float('inf')

        print(f"\nRatios (OMRAT/IWRAP):")
        print(f"  Allision: {allision_ratio:.2f}x")
        print(f"  Grounding: {grounding_ratio:.2f}x")

        # Check if within tolerance
        lower_bound = 1 - self.TOLERANCE
        upper_bound = 1 + self.TOLERANCE

        allision_ok = lower_bound <= allision_ratio <= upper_bound
        grounding_ok = lower_bound <= grounding_ratio <= upper_bound

        print(f"\nWithin ±50% tolerance:")
        print(f"  Allision: {'YES' if allision_ok else 'NO'} ({lower_bound:.1f}x to {upper_bound:.1f}x)")
        print(f"  Grounding: {'YES' if grounding_ok else 'NO'} ({lower_bound:.1f}x to {upper_bound:.1f}x)")

        # Store results for further analysis
        self.last_allision = allision
        self.last_grounding = grounding

        # Assert that results are within tolerance
        # This ensures the IWRAP calibration stays valid
        assert allision_ok, f"Allision ratio {allision_ratio:.2f}x not within ±50% tolerance"
        assert grounding_ok, f"Grounding ratio {grounding_ratio:.2f}x not within ±50% tolerance"


class TestBinningStrategy:
    """Test different binning strategies to reduce computation."""

    @pytest.fixture
    def project_data(self):
        """Load the example project data."""
        data_path = Path(__file__).parent / 'example_data' / 'proj.omrat'
        return load_omrat_data(str(data_path))

    def test_compute_effective_bins(self, project_data):
        """Compute the effective number of unique calculations needed."""
        info = analyze_data_structure(project_data)

        depth_contours = info['depth_contours']
        draughts = info['unique_draughts']
        heights = info['unique_heights']
        structure_heights = info['structure_heights']

        print("\n=== Computation Reduction Analysis ===")
        print(f"Without binning:")
        print(f"  Unique draughts: {len(draughts)}")
        print(f"  Unique heights: {len(heights)}")
        print(f"  Total unique ship configurations: {len(draughts) * len(heights)}")

        # For grounding, we only need to compute for unique depth thresholds
        # that actually affect ships
        effective_depth_bins = 0
        for depth in depth_contours:
            affected = sum(1 for d in draughts if d > depth)
            if affected > 0:
                effective_depth_bins += 1

        # For allision, we only need to compute for unique structure heights
        effective_height_bins = 0
        for struct_h in structure_heights:
            affected = sum(1 for h in heights if h < struct_h)
            if affected > 0:
                effective_height_bins += 1

        print(f"\nWith smart binning:")
        print(f"  Effective depth bins: {effective_depth_bins}")
        print(f"  Effective height bins: {effective_height_bins}")

        reduction = 1 - (effective_depth_bins / len(draughts))
        print(f"\nComputation reduction for grounding: {reduction*100:.1f}%")

        # The key insight: we only need ONE corridor calculation per depth bin
        # because the corridor shape only depends on the depth threshold, not
        # individual ship draughts.


if __name__ == '__main__':
    # Run with verbose output
    pytest.main([__file__, '-v', '-s'])
