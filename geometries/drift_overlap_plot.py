"""Matplotlib bottom-panel plot for the drifting-overlap visualiser.

The :func:`visualize` function paints the **bottom axis** of the
overlap dialog: the weighted-PDF curve, the failure-remains
:math:`P_{NR}` curve, and the green "intersection extent" axvspan,
plus a picture-in-picture inset that zooms into the intersection
band so the curves are readable when the obstacle sits far down the
drift direction.

Pure matplotlib -- no Qt, no QGIS.  Tested via
``tests/test_get_drifting_overlap.py``.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from compute.basic_equations import get_not_repaired


# 50 000 km -- larger than Earth's circumference.  Distances above this
# are numerical blow-ups from convex-hull unions on degenerate
# polygons; they squash the real data into invisibility on the x-axis.
_SANE_DIST_M = 5e7


def _filter_finite_distances(distances: np.ndarray) -> np.ndarray:
    finite = distances[np.isfinite(distances)]
    return finite[(finite >= 0) & (finite < _SANE_DIST_M)]


def _intersection_extent(finite: np.ndarray) -> tuple[float, float]:
    """5th/95th-percentile extent that drops near-leg edge artifacts."""
    if finite.size >= 4:
        return (
            float(np.percentile(finite, 5)),
            float(np.percentile(finite, 95)),
        )
    return float(finite.min()), float(finite.max())


def _normalised_weights(weights: list[float]) -> np.ndarray:
    arr = np.array(weights, dtype=float)
    if arr.sum() > 0:
        arr = arr / arr.sum()
    return arr


def _distribution_x_max(
    distributions: list[Any],
    intersection_max: float,
) -> float:
    """Right edge of the main axis x-range (always finite, > 0)."""
    dist_max = 0.0
    for dist in distributions:
        try:
            mean_v = float(dist.mean())
            std_v = float(dist.std())
        except Exception:
            continue
        if not (np.isfinite(mean_v) and np.isfinite(std_v)):
            continue
        dist_max = max(dist_max, abs(mean_v) + 4.0 * std_v)
    x_max = max(intersection_max * 1.2, dist_max, 1.0)
    if not np.isfinite(x_max) or x_max <= 0:
        x_max = max(
            intersection_max if np.isfinite(intersection_max) else 0.0,
            1.0,
        )
    if not np.isfinite(x_max) or x_max <= 0:
        x_max = 1.0
    return x_max


def _combined_pdf(
    x: np.ndarray,
    distributions: list[Any],
    weights: np.ndarray,
) -> np.ndarray:
    pdf = np.zeros_like(x)
    for dist, weight in zip(distributions, weights):
        try:
            pdf = pdf + float(weight) * np.asarray(dist.pdf(x))
        except Exception:
            continue
    return pdf


def _failure_remains(x: np.ndarray, data: dict[str, Any]) -> np.ndarray:
    return np.array([
        get_not_repaired(data['drift']['repair'], data['drift']['speed'], x_)
        for x_ in x
    ])


def _show_no_data(ax: Axes, msg: str) -> None:
    ax.text(
        0.5, 0.5, msg,
        transform=ax.transAxes, ha='center', va='center',
        fontsize=10, color='gray',
    )
    ax.figure.canvas.draw()


def _clear_existing_insets(fig) -> None:
    for art in list(getattr(fig, 'axes', [])):
        if getattr(art, '_omrat_inset', False):
            try:
                art.remove()
            except Exception:
                pass


def _draw_main_curves(
    ax: Axes,
    x: np.ndarray,
    pdf: np.ndarray,
    fail: np.ndarray,
    span_lo: float,
    span_hi: float,
):
    pdf_peak = float(np.max(pdf)) if pdf.size else 0.0
    if pdf_peak > 0:
        pdf_normalized = pdf / pdf_peak
        pdf_label = (
            f"Prob. leg overlap (PDF, normalized; peak={pdf_peak:.2e})"
        )
    else:
        pdf_normalized = pdf
        pdf_label = "Prob. leg overlap (PDF, all-zero)"

    line_pdf, = ax.plot(
        x, pdf_normalized, color='blue', label=pdf_label,
    )
    line_fail, = ax.plot(
        x, fail, color='red', label='Prob. failure remains',
    )
    span = ax.axvspan(
        span_lo, span_hi, color='green', alpha=0.3,
        label='Intersection Extent',
    )
    return line_pdf, line_fail, span, pdf_peak


def _widen_zero_span(
    intersection_min: float, intersection_max: float, x_max: float,
) -> tuple[float, float]:
    """Widen a degenerate axvspan so it stays visible on screen."""
    span_lo = intersection_min
    span_hi = intersection_max
    if span_hi - span_lo < x_max * 0.005:
        eps = max(x_max * 0.005, 1.0)
        span_lo = max(0.0, intersection_min - eps / 2)
        span_hi = intersection_max + eps / 2
    return span_lo, span_hi


def _draw_zoom_inset(
    ax: Axes,
    distributions: list[Any],
    weights: np.ndarray,
    pdf_peak: float,
    intersection_min: float,
    intersection_max: float,
    span_lo: float,
    span_hi: float,
    x_max: float,
    data: dict[str, Any],
) -> None:
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset = inset_axes(
            ax, width="38%", height="50%", loc='center right',
            borderpad=1.0,
        )
        inset._omrat_inset = True  # type: ignore[attr-defined]

        pad = max(
            (intersection_max - intersection_min) * 0.5,
            x_max * 0.005, 50.0,
        )
        x_lo = max(0.0, intersection_min - pad)
        x_hi = intersection_max + pad
        x_zoom = np.linspace(x_lo, x_hi, 400)
        pdf_zoom = _combined_pdf(x_zoom, distributions, weights)
        if pdf_peak > 0:
            pdf_zoom = pdf_zoom / pdf_peak
        fail_zoom = _failure_remains(x_zoom, data)
        inset.plot(x_zoom, pdf_zoom, color='blue', linewidth=1.0)
        inset.plot(x_zoom, fail_zoom, color='red', linewidth=1.0)
        inset.axvspan(span_lo, span_hi, color='green', alpha=0.3)
        inset.set_xlim(x_lo, x_hi)
        try:
            fail_at_span_lo = float(get_not_repaired(
                data['drift']['repair'],
                data['drift']['speed'],
                span_lo,
            ))
        except Exception:
            fail_at_span_lo = 1.0
        pdf_peak_zoom = float(pdf_zoom.max()) if pdf_zoom.size else 0.0
        y_top = max(fail_at_span_lo, pdf_peak_zoom) * 1.1
        if not np.isfinite(y_top) or y_top <= 0:
            y_top = 1.05
        inset.set_ylim(0, min(y_top, 1.05))
        inset.set_title("Overlap zoom", fontsize=8)
        inset.tick_params(labelsize=6)
        inset.grid(True, alpha=0.2)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).debug(
            f"Could not render overlap-zoom inset: {exc}"
        )


def visualize(
    ax3: Axes,
    distances: np.ndarray,
    distributions: list[Any],
    weights: list[float],
    weighted_overlap: float,
    data: dict[Any],
) -> None:
    """Paint the bottom panel: PDF + failure-remains + intersection extent.

    The original signature is kept so existing callers keep working.
    """
    ax3.clear()
    if weighted_overlap is None:
        return
    _clear_existing_insets(ax3.figure)

    if distances is None or getattr(distances, 'size', 0) == 0:
        _show_no_data(
            ax3,
            "No intersection between this drift direction\n"
            "and any object/depth -- nothing to plot.",
        )
        return

    finite_dists = _filter_finite_distances(distances)
    if finite_dists.size == 0:
        _show_no_data(
            ax3,
            "All intersection distances are non-finite or out of bounds\n"
            "-- nothing to plot.",
        )
        return

    intersection_min, intersection_max = _intersection_extent(finite_dists)
    weights_arr = _normalised_weights(weights)
    x_max = _distribution_x_max(distributions, intersection_max)

    x = np.linspace(0, x_max, 800)
    combined_pdf = _combined_pdf(x, distributions, weights_arr)
    not_repaireds = _failure_remains(x, data)

    span_lo, span_hi = _widen_zero_span(
        intersection_min, intersection_max, x_max,
    )

    line_pdf, line_fail, span, pdf_peak = _draw_main_curves(
        ax3, x, combined_pdf, not_repaireds, span_lo, span_hi,
    )

    ax3.set_xlabel("Distance from Closest Point (m)")
    ax3.set_ylabel("Probability (0..1, PDF normalized to peak)")
    ax3.set_xlim(0, x_max)
    ax3.set_ylim(0, 1.05)
    ax3.legend(
        handles=[line_pdf, line_fail, span],
        loc='upper right', fontsize=8,
    )

    _draw_zoom_inset(
        ax3, distributions, weights_arr, pdf_peak,
        intersection_min, intersection_max, span_lo, span_hi, x_max, data,
    )

    plt.suptitle("Interactive Visualization")
    ax3.figure.canvas.draw()
