# -*- coding: utf-8 -*-
"""
Statistical distribution calculations for drift corridor parameters.

Calculates projection distance based on repair time probability
and distribution width based on lateral deviation.
"""

import numpy as np
from scipy import stats


def get_projection_distance(repair_params: dict, drift_speed_ms: float,
                            target_prob: float = 1e-3,
                            max_distance: float = 50000) -> float:
    """
    Calculate the projection distance where prob_not_repaired drops to target_prob.

    Uses a lognormal distribution for repair time to determine how far
    a drifting ship might travel before repairs are completed.

    Args:
        repair_params: Lognormal distribution parameters:
            - std: Shape parameter (sigma)
            - loc: Location parameter
            - scale: Scale parameter (exp(mu))
            - use_lognormal: Whether to use lognormal (default True)
        drift_speed_ms: Drift speed in m/s
        target_prob: Target probability for P(not repaired), default 1e-3 (0.1%)
        max_distance: Maximum distance cap in meters, default 50km

    Returns:
        Projection distance in meters (capped at max_distance)
    """
    if drift_speed_ms <= 0 or drift_speed_ms > 10:
        drift_speed_ms = 1.0

    if repair_params.get('use_lognormal', True):
        try:
            std_val = repair_params.get('std', 0.95)
            loc_val = repair_params.get('loc', 0.2)
            scale_val = repair_params.get('scale', 0.85)

            if std_val <= 0 or scale_val <= 0:
                return min(10000, max_distance)

            dist = stats.lognorm(std_val, loc_val, scale_val)
            cdf_target = min(1 - target_prob, 0.999)
            drift_time_hours = dist.ppf(cdf_target)

            if drift_time_hours > 48 or drift_time_hours < 0 or not np.isfinite(drift_time_hours):
                return min(10000, max_distance)

            distance_m = drift_time_hours * 3600 * drift_speed_ms

            if not np.isfinite(distance_m) or distance_m < 0:
                return min(10000, max_distance)

            return min(distance_m, max_distance)
        except Exception:
            return min(10000, max_distance)
    else:
        return min(10000, max_distance)


def get_distribution_width(std: float, coverage: float = 0.99) -> float:
    """
    Calculate the width that covers 'coverage' fraction of a normal distribution.

    For 99% coverage: width = 2 * Z_0.995 * std ~ 2 * 2.576 * std

    Args:
        std: Standard deviation of the lateral distribution in meters
        coverage: Fraction of distribution to cover (default 0.99 = 99%)

    Returns:
        Total width in meters that covers the specified fraction
    """
    z = stats.norm.ppf((1 + coverage) / 2)
    return 2 * z * std
