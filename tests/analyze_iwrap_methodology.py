# -*- coding: utf-8 -*-
"""
Analyze IWRAP methodology to understand the probability calculation differences.

This script:
1. Loads the IWRAP detail files to understand their structure
2. Analyzes what factors are included in IWRAP's calculations
3. Proposes corrections to OMRAT's methodology
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add parent directory to path
script_dir = Path(__file__).parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))


def load_iwrap_grounding_details(filepath: str) -> list[dict]:
    """Parse the IWRAP grounding details file."""
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse the format: leg_id, depth_id, contribution
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('[') or line.startswith('---'):
            continue

        # Format: 4,UUID1	5,UUID2	probability
        parts = line.split('\t')
        if len(parts) >= 3:
            try:
                prob = float(parts[2])
                entries.append({
                    'leg_info': parts[0],
                    'depth_info': parts[1],
                    'probability': prob
                })
            except (ValueError, IndexError):
                continue

    return entries


def load_iwrap_allision_details(filepath: str) -> list[dict]:
    """Parse the IWRAP allision details file."""
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse the format
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('[') or line.startswith('CAT') or line.startswith('LEG') or line.startswith('---'):
            continue

        parts = line.split('\t')
        if len(parts) >= 3:
            try:
                prob = float(parts[2])
                entries.append({
                    'structure_info': parts[0],
                    'leg_info': parts[1],
                    'probability': prob
                })
            except (ValueError, IndexError):
                continue

    return entries


def analyze_iwrap_grounding(entries: list[dict]) -> dict:
    """Analyze IWRAP grounding probability distribution."""
    probs = [e['probability'] for e in entries]

    total = sum(probs)
    min_prob = min(probs)
    max_prob = max(probs)
    median_prob = np.median(probs)

    # Count by order of magnitude
    magnitude_counts = defaultdict(int)
    for p in probs:
        mag = int(np.floor(np.log10(p))) if p > 0 else -999
        magnitude_counts[mag] += 1

    return {
        'count': len(entries),
        'total': total,
        'min': min_prob,
        'max': max_prob,
        'median': median_prob,
        'magnitude_distribution': dict(sorted(magnitude_counts.items()))
    }


def analyze_iwrap_allision(entries: list[dict]) -> dict:
    """Analyze IWRAP allision probability distribution."""
    probs = [e['probability'] for e in entries]

    total = sum(probs)
    min_prob = min(probs) if probs else 0
    max_prob = max(probs) if probs else 0
    median_prob = np.median(probs) if probs else 0

    # Count by order of magnitude
    magnitude_counts = defaultdict(int)
    for p in probs:
        if p > 0:
            mag = int(np.floor(np.log10(p)))
            magnitude_counts[mag] += 1

    return {
        'count': len(entries),
        'total': total,
        'min': min_prob,
        'max': max_prob,
        'median': median_prob,
        'magnitude_distribution': dict(sorted(magnitude_counts.items()))
    }


def load_omrat_data(filepath: str) -> dict:
    """Load .omrat project file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_theoretical_grounding_comparison(data: dict) -> None:
    """
    Compare theoretical grounding calculation approaches.

    IWRAP approach (hypothesized):
    - P(grounding) = Σ over all (ship, depth) pairs:
        P(blackout) × P(drift_to_depth) × P(not_repaired) × P(not_anchored)

    Where P(drift_to_depth) is computed as:
    - For each ship position along the leg
    - The probability of drifting in a specific direction
    - AND arriving at the depth contour
    - This is integrated over the entire leg and lateral distribution

    The key difference from OMRAT:
    - IWRAP likely integrates P(drift) = exp(-d/λ) where λ is a characteristic distance
    - This makes distant obstacles contribute exponentially less
    """
    print("\n" + "="*70)
    print("THEORETICAL GROUNDING CALCULATION COMPARISON")
    print("="*70)

    drift = data.get('drift', {})
    drift_p = float(drift.get('drift_p', 0.0))
    anchor_p = float(drift.get('anchor_p', 0.0))
    anchor_d = float(drift.get('anchor_d', 0.0))
    drift_speed_kts = float(drift.get('speed', 0.0))
    drift_speed_ms = drift_speed_kts * 1852.0 / 3600.0

    blackout_per_hour = drift_p / (365.0 * 24.0)

    print(f"\nDrift parameters:")
    print(f"  drift_p (per year) = {drift_p}")
    print(f"  blackout_per_hour = {blackout_per_hour:.6e}")
    print(f"  drift_speed = {drift_speed_ms:.3f} m/s")
    print(f"  anchor_p = {anchor_p}")
    print(f"  anchor_d = {anchor_d}")

    # Repair distribution parameters
    repair = drift.get('repair', {})
    from scipy import stats

    if repair.get('use_lognormal', False):
        repair_dist = stats.lognorm(repair['std'], repair['loc'], repair['scale'])

        # Calculate P(not repaired) for various drift times
        print(f"\nRepair probability (lognormal distribution):")
        print(f"  Parameters: std={repair['std']}, loc={repair['loc']}, scale={repair['scale']}")
        print(f"\n  P(not repaired) at different distances (at {drift_speed_ms:.3f} m/s):")

        for dist_m in [100, 500, 1000, 2000, 5000, 10000, 20000]:
            drift_time_hours = (dist_m / drift_speed_ms) / 3600
            p_nr = 1 - repair_dist.cdf(drift_time_hours)
            print(f"    {dist_m:6d}m ({drift_time_hours:.2f}h): P(not repaired) = {p_nr:.6f}")

    # Rose distribution
    rose_vals = {int(k): float(v) for k, v in drift.get('rose', {}).items()}
    rose_total = sum(rose_vals.values())

    print(f"\nRose distribution (wind/current direction):")
    for angle in sorted(rose_vals.keys()):
        prob = rose_vals[angle] / rose_total if rose_total > 0 else 0
        print(f"  {angle:3d}°: {prob:.4f} ({rose_vals[angle]:.2f}%)")

    # Key insight: IWRAP's tiny grounding probabilities
    print("\n" + "="*70)
    print("KEY INSIGHT: WHY IWRAP GROUNDING IS SO SMALL")
    print("="*70)

    print("""
IWRAP grounding contributions are in the range 1e-11 to 1e-20.
This is because IWRAP calculates:

  P(grounding at specific depth) = ∫∫ P(ship at x,y) × P(drift to depth | x,y) × P(not repaired) × P(not anchored) dx dy

Where P(drift to depth | x,y) is VERY small because:
1. Only ships in a narrow corridor can reach the depth contour
2. The probability of reaching the depth contour decreases with distance
3. Only one direction (out of 8) leads to the depth contour

OMRAT's current approach:
- Uses hole_pct = overlap_area / corridor_area ≈ 0.01 to 0.1
- This is a GEOMETRIC probability, not weighted by position

The fix needed:
- OMRAT's hole_pct should be multiplied by:
  1. The fraction of the leg that can actually drift to the obstacle
  2. An exponential decay factor based on distance

Alternative interpretation:
- IWRAP may use a different base probability model
- The per-ship, per-obstacle contribution should be ≈ 1e-12 to 1e-14
- Summing over all ships and obstacles gives the final total
""")


def estimate_correction_factor(omrat_grounding: float, iwrap_grounding: float) -> float:
    """Estimate the correction factor needed."""
    if iwrap_grounding > 0:
        return iwrap_grounding / omrat_grounding
    return 0.0


def main():
    """Main entry point."""
    # Load IWRAP detail files
    grounding_path = script_dir / 'example_data' / 'grounding_iwrap_details.csv'
    allision_path = script_dir / 'example_data' / 'allision_iwrap_details.csv'
    data_path = script_dir / 'example_data' / 'proj.omrat'

    print("="*70)
    print("IWRAP METHODOLOGY ANALYSIS")
    print("="*70)

    # Analyze IWRAP grounding
    print("\n" + "="*70)
    print("IWRAP GROUNDING ANALYSIS")
    print("="*70)

    if grounding_path.exists():
        grounding_entries = load_iwrap_grounding_details(str(grounding_path))
        analysis = analyze_iwrap_grounding(grounding_entries)

        print(f"\nNumber of (leg, depth) contributions: {analysis['count']}")
        print(f"Total grounding probability: {analysis['total']:.6e}")
        print(f"Min contribution: {analysis['min']:.6e}")
        print(f"Max contribution: {analysis['max']:.6e}")
        print(f"Median contribution: {analysis['median']:.6e}")

        print(f"\nDistribution by order of magnitude:")
        for mag, count in analysis['magnitude_distribution'].items():
            print(f"  10^{mag}: {count} entries")
    else:
        print(f"Grounding file not found: {grounding_path}")

    # Analyze IWRAP allision
    print("\n" + "="*70)
    print("IWRAP ALLISION ANALYSIS")
    print("="*70)

    if allision_path.exists():
        allision_entries = load_iwrap_allision_details(str(allision_path))
        analysis = analyze_iwrap_allision(allision_entries)

        print(f"\nNumber of (structure, leg) contributions: {analysis['count']}")
        print(f"Total allision probability: {analysis['total']:.6e}")
        print(f"Min contribution: {analysis['min']:.6e}")
        print(f"Max contribution: {analysis['max']:.6e}")
        print(f"Median contribution: {analysis['median']:.6e}")

        print(f"\nDistribution by order of magnitude:")
        for mag, count in analysis['magnitude_distribution'].items():
            print(f"  10^{mag}: {count} entries")
    else:
        print(f"Allision file not found: {allision_path}")

    # Load OMRAT data for comparison
    if data_path.exists():
        data = load_omrat_data(str(data_path))
        compute_theoretical_grounding_comparison(data)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: REQUIRED CHANGES")
    print("="*70)

    print("""
1. GROUNDING PROBABILITY (26,000x too high):
   - OMRAT hole_pct values are ~1e-5 to 1e-7
   - IWRAP individual contributions are ~1e-11 to 1e-17
   - Difference: ~10^5 to 10^10 times

   Possible causes:
   a) OMRAT doesn't weight by distance (exponential decay missing)
   b) OMRAT counts each polygon fragment separately
   c) OMRAT may double-count across directions

   Recommended fix:
   - Add exponential decay: P(hit) = P(geometric) × exp(-distance/λ)
   - Where λ is calibrated to match IWRAP results
   - OR use a completely different methodology based on actual drift simulation

2. ALLISION PROBABILITY (5.8x too low):
   - OMRAT: 6.07e-03
   - IWRAP: 3.51e-02
   - Ratio: 0.17x (should be 1.0x ± 50%)

   This suggests OMRAT is under-counting allision events.
   Possible causes:
   a) PDF correction factor may be wrong
   b) Structure heights may not match IWRAP
   c) Different ship height distributions
""")

    return 0


if __name__ == '__main__':
    sys.exit(main())
