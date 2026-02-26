# -*- coding: utf-8 -*-
"""
Standalone script to compare OMRAT calculation results with IWRAP.

Run this script from the OMRAT directory:
    python tests/run_iwrap_comparison.py

IWRAP reference values:
- Grounding probability: 1.62e-07
- Allision probability: 0.0351
"""

import json
import sys
import os
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))


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
    info['drift_speed'] = drift.get('speed', 0)  # Key is 'speed', not 'drift_v'
    info['rose'] = drift.get('rose', {})
    info['repair'] = drift.get('repair', {})

    return info


def create_smart_bins(draughts: list[float], depth_contours: list[float]) -> dict[float, list[float]]:
    """
    Create smart bins mapping depth thresholds to lists of draughts.

    Ships only ground where their draught > water depth.
    So we only need to calculate separately for each unique depth contour.
    """
    bins = {}
    for depth in depth_contours:
        affected_draughts = [d for d in draughts if d > depth]
        if affected_draughts:
            bins[depth] = affected_draughts
    return bins


def print_data_analysis(data: dict) -> None:
    """Print detailed data analysis."""
    info = analyze_data_structure(data)

    print("\n" + "="*60)
    print("DATA STRUCTURE ANALYSIS")
    print("="*60)
    print(f"\nDepth contours: {info['depth_contours']}")
    print(f"Number of depth entries: {info['num_depths']}")
    print(f"Number of structures: {info['num_structures']}")
    print(f"Structure heights: {info['structure_heights']}")
    print(f"Number of traffic cells: {info['num_cells']}")
    print(f"Unique draughts: {info['num_unique_draughts']}")
    print(f"Unique heights: {info['num_unique_heights']}")
    print(f"Total ships/year: {info['total_ships_per_year']}")
    print(f"Drift probability (per hour): {info['drift_prob']}")
    print(f"Anchor probability: {info['anchor_prob']}")
    print(f"Anchor depth factor: {info['anchor_depth']} (times draught)")
    print(f"Drift speed: {info['drift_speed']:.2f} knots")
    print(f"Wind rose: {info['rose']}")
    print(f"Repair params: {info['repair']}")

    # Smart binning analysis
    depth_bins = create_smart_bins(info['unique_draughts'], info['depth_contours'])

    print("\n" + "="*60)
    print("SMART BINNING ANALYSIS")
    print("="*60)
    print("\nDepth bins (ships that would ground at each depth contour):")
    for depth, draughts in sorted(depth_bins.items()):
        print(f"  Depth <= {depth}m: {len(draughts)} unique draughts affected")

    print(f"\nWithout binning: {info['num_unique_draughts']} unique draughts")
    print(f"With binning: {len(depth_bins)} effective depth bins")
    reduction = 1 - (len(depth_bins) / info['num_unique_draughts'])
    print(f"Computation reduction: {reduction*100:.1f}%")

    return info


def run_calculation_comparison(data: dict, debug: bool = True) -> tuple[float, float]:
    """
    Run the OMRAT calculation and compare with IWRAP values.

    Returns (allision, grounding) probabilities.
    """
    # Mock the QGIS-specific parts
    from unittest.mock import MagicMock, patch

    # Create mock UI elements
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

    # Import and run the calculation
    from compute.run_calculations import Calculation

    calc = Calculation(mock_parent)

    if debug:
        # Print intermediate values before calculation
        drift = data.get('drift', {})
        drift_p = float(drift.get('drift_p', 0.0))
        blackout_per_hour = drift_p / (365.0 * 24.0)
        print(f"\n  Debug: drift_p = {drift_p}")
        print(f"  Debug: blackout_per_hour = {blackout_per_hour:.6e}")

        drift_speed_kts = float(drift.get('speed', 0.0))
        drift_speed_ms = drift_speed_kts * 1852.0 / 3600.0
        print(f"  Debug: drift_speed = {drift_speed_kts:.2f} knots = {drift_speed_ms:.3f} m/s")

        # Repair parameters
        repair = drift.get('repair', {})
        print(f"  Debug: repair params = {repair}")

    # Get intermediate values before full calculation
    from compute.run_calculations import clean_traffic, split_structures_and_depths

    # Build transformed data to see probability holes
    # (This replicates some of the calculation setup)
    structures, depths = split_structures_and_depths(data)
    print(f"\n  Structures after split: {len(structures)}")
    print(f"  Depths after split: {len(depths)}")

    # Check sample depth entries
    depth_by_val = {}
    for d in depths:
        dval = d['depth']
        depth_by_val.setdefault(dval, []).append(d['id'])
    print(f"\n  Depth fragments by depth value:")
    for dval in sorted(depth_by_val.keys())[:6]:
        print(f"    {dval}m: {len(depth_by_val[dval])} fragments")

    allision, grounding = calc.run_drifting_model(data)

    # Debug: Check if the calculation has intermediate data
    if debug:
        print("\n  Note: The depth aggregation combines fragments into single obstacles per depth value.")
        print("  This should reduce the grounding probability significantly.")

    # Try to get detailed breakdown
    if debug:
        # Get the report for detailed analysis
        report = getattr(calc, 'drifting_report', None)
        if report:
            totals = report.get('totals', {})
            print(f"\n  Report totals: allision={totals.get('allision', 0):.3e}, grounding={totals.get('grounding', 0):.3e}")

            # Count unique depth contours contributing
            by_obj = report.get('by_object', {})
            depths_contributing = [k for k in by_obj.keys() if k.startswith('Depth')]
            structs_contributing = [k for k in by_obj.keys() if k.startswith('Structure')]
            print(f"  Contributing depths: {len(depths_contributing)}")
            print(f"  Contributing structures: {len(structs_contributing)}")

            # Sum by depth value
            depth_sums = {}
            for k, v in by_obj.items():
                if k.startswith('Depth'):
                    # Extract depth value from name like "Depth - 1_15_0"
                    # The depth index (1, 2, 3, etc.) corresponds to depth contour
                    parts = k.split(' - ')[1].split('_')
                    depth_idx = parts[0]  # First part is the depth index
                    grounding_val = v.get('grounding', 0) if isinstance(v, dict) else 0
                    depth_sums[depth_idx] = depth_sums.get(depth_idx, 0) + grounding_val

            print(f"\n  Grounding by depth contour index:")
            for idx in sorted(depth_sums.keys(), key=int):
                print(f"    Depth index {idx}: {depth_sums[idx]:.3e}")

    if debug and hasattr(calc, 'drifting_report'):
        report = calc.drifting_report
        if report:
            # Print some details from the report
            by_obj = report.get('by_object', {})
            print(f"\n  By object breakdown:")
            for obj_name, obj_data in by_obj.items():
                if isinstance(obj_data, dict):
                    allision_val = obj_data.get('allision', 0)
                    grounding_val = obj_data.get('grounding', 0)
                    print(f"    {obj_name}: allision={allision_val:.3e}, grounding={grounding_val:.3e}")
                else:
                    print(f"    {obj_name}: {obj_data}")

    return allision, grounding


def trace_single_ship_calculation(data: dict) -> None:
    """Trace through a single ship's grounding calculation to understand the math."""
    print("\n" + "="*60)
    print("TRACING SINGLE SHIP CALCULATION")
    print("="*60)

    drift = data.get('drift', {})
    drift_p = float(drift.get('drift_p', 0.0))
    anchor_p = float(drift.get('anchor_p', 0.0))
    anchor_d = float(drift.get('anchor_d', 0.0))
    drift_speed_kts = float(drift.get('speed', 0.0))
    drift_speed_ms = drift_speed_kts * 1852.0 / 3600.0

    blackout_per_hour = drift_p / (365.0 * 24.0)

    # Pick a typical ship
    draught = 5.0  # meters
    height = 8.0  # meters (can hit 10m structures)
    speed_kts = 10.0
    freq = 100  # ships/year
    line_length = 10000  # meters (10 km leg)

    anchor_threshold = anchor_d * draught

    print(f"\nShip parameters:")
    print(f"  Draught: {draught}m")
    print(f"  Height: {height}m")
    print(f"  Speed: {speed_kts} knots")
    print(f"  Frequency: {freq} ships/year")
    print(f"  Line length: {line_length}m")
    print(f"  Anchor threshold: {anchor_threshold}m (7 × draught)")

    print(f"\nDrift parameters:")
    print(f"  drift_p = {drift_p}")
    print(f"  blackout_per_hour = {blackout_per_hour:.6e}")
    print(f"  anchor_p = {anchor_p}")
    print(f"  drift_speed = {drift_speed_ms:.3f} m/s")

    # Calculate hours present
    hours_present = (line_length / (speed_kts * 1852.0)) * freq
    base = hours_present * blackout_per_hour
    print(f"\nExposure calculation:")
    print(f"  hours_present (total for {freq} ships) = {hours_present:.4f} hours")
    print(f"  base = hours_present × blackout_per_hour = {base:.6e}")

    # Rose probability (1/8 for uniform)
    rp = 0.125
    print(f"\nRose probability (one direction): {rp}")

    # Simulate hitting a 3m depth contour
    depth = 3.0
    dist = 1000  # meters to depth contour
    hole_pct = 0.1  # 10% chance of hitting this depth

    print(f"\nSimulated depth obstacle:")
    print(f"  Depth value: {depth}m (< draught {draught}m, so grounding possible)")
    print(f"  Distance: {dist}m")
    print(f"  hole_pct: {hole_pct}")

    # Repair probability
    from scipy import stats
    repair_params = drift.get('repair', {})
    drift_time_hours = (dist / drift_speed_ms) / 3600
    repair_dist = stats.lognorm(repair_params['std'], repair_params['loc'], repair_params['scale'])
    p_nr = 1 - repair_dist.cdf(drift_time_hours)
    print(f"\nRepair calculation:")
    print(f"  Drift time to obstacle: {drift_time_hours:.4f} hours")
    print(f"  P(not repaired) = {p_nr:.4f}")

    # Anchoring first
    remaining_prob = 1.0
    if depth < anchor_threshold:
        remaining_prob *= (1.0 - anchor_p * hole_pct)
        print(f"\nAnchoring (depth {depth}m < threshold {anchor_threshold}m):")
        print(f"  remaining_prob *= (1 - {anchor_p} × {hole_pct}) = {remaining_prob:.4f}")

    # Grounding contribution
    if depth < draught:
        contrib = base * rp * remaining_prob * hole_pct * p_nr
        print(f"\nGrounding contribution:")
        print(f"  contrib = base × rp × remaining_prob × hole_pct × p_nr")
        print(f"  contrib = {base:.6e} × {rp} × {remaining_prob:.4f} × {hole_pct} × {p_nr:.4f}")
        print(f"  contrib = {contrib:.6e}")

    # Compare to IWRAP expectation
    print(f"\nComparison:")
    print(f"  OMRAT single contribution: {contrib:.6e}")
    print(f"  IWRAP typical contribution: ~1e-12 to 1e-14")
    print(f"  Ratio: ~{contrib / 1e-13:.0f}x")


def main():
    """Main entry point."""
    # Load the example data
    data_path = script_dir / 'example_data' / 'proj.omrat'

    if not data_path.exists():
        print(f"ERROR: Project file not found: {data_path}")
        return 1

    print(f"Loading project: {data_path}")
    data = load_omrat_data(str(data_path))

    # Analyze data structure
    info = print_data_analysis(data)

    # Trace single ship calculation
    trace_single_ship_calculation(data)

    # IWRAP reference values
    IWRAP_GROUNDING = 1.62e-07
    IWRAP_ALLISION = 0.0351

    print("\n" + "="*60)
    print("IWRAP REFERENCE VALUES")
    print("="*60)
    print(f"Grounding probability: {IWRAP_GROUNDING:.3e}")
    print(f"Allision probability: {IWRAP_ALLISION:.3e}")

    # Try to run the calculation
    print("\n" + "="*60)
    print("RUNNING OMRAT CALCULATION")
    print("="*60)

    # Test with adjusted drift_p
    print("\n=== TEST 1: Original drift_p ===")
    try:
        allision, grounding = run_calculation_comparison(data, debug=False)
        print(f"  With drift_p={data['drift']['drift_p']}: allision={allision:.3e}, grounding={grounding:.3e}")

        # Test with lower drift_p
        print("\n=== TEST 2: Adjusted drift_p to 0.1 ===")
        data_modified = json.loads(json.dumps(data))  # Deep copy
        data_modified['drift']['drift_p'] = 0.1  # 0.1 blackouts per year
        allision2, grounding2 = run_calculation_comparison(data_modified, debug=False)
        print(f"  With drift_p=0.1: allision={allision2:.3e}, grounding={grounding2:.3e}")

        # Test with even lower drift_p
        print("\n=== TEST 3: Adjusted drift_p to 0.01 ===")
        data_modified['drift']['drift_p'] = 0.01  # 0.01 blackouts per year
        allision3, grounding3 = run_calculation_comparison(data_modified, debug=False)
        print(f"  With drift_p=0.01: allision={allision3:.3e}, grounding={grounding3:.3e}")

    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== FINAL COMPARISON ===")
    try:
        allision, grounding = run_calculation_comparison(data, debug=True)

        print(f"\nOMRAT Results:")
        print(f"  Allision probability: {allision:.3e}")
        print(f"  Grounding probability: {grounding:.3e}")

        # Calculate ratios
        allision_ratio = allision / IWRAP_ALLISION if IWRAP_ALLISION > 0 else float('inf')
        grounding_ratio = grounding / IWRAP_GROUNDING if IWRAP_GROUNDING > 0 else float('inf')

        print(f"\nRatios (OMRAT/IWRAP):")
        print(f"  Allision: {allision_ratio:.2f}x")
        print(f"  Grounding: {grounding_ratio:.2f}x")

        # Check if within tolerance
        TOLERANCE = 0.5
        lower_bound = 1 - TOLERANCE
        upper_bound = 1 + TOLERANCE

        allision_ok = lower_bound <= allision_ratio <= upper_bound
        grounding_ok = lower_bound <= grounding_ratio <= upper_bound

        print(f"\nWithin ±50% tolerance:")
        print(f"  Allision: {'✓ YES' if allision_ok else '✗ NO'} (target: {lower_bound:.1f}x to {upper_bound:.1f}x)")
        print(f"  Grounding: {'✓ YES' if grounding_ok else '✗ NO'} (target: {lower_bound:.1f}x to {upper_bound:.1f}x)")

        if allision_ok and grounding_ok:
            print("\n✓ SUCCESS: Both values within tolerance!")
            return 0
        else:
            print("\n✗ Values need adjustment")
            return 1

    except ImportError as e:
        print(f"\nCould not run calculation due to missing dependencies: {e}")
        print("The data analysis above shows the structure of the project.")
        print("\nTo run the full calculation, use the QGIS environment with pytest-qgis.")
        return 2
    except Exception as e:
        print(f"\nError running calculation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
