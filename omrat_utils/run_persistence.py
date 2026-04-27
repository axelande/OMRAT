"""Write a finished OMRAT run's spatial result layers into a single
per-run GeoPackage, and load them back for the "Add results to map"
button.

The master history database (see :mod:`omrat_utils.run_history`) only
stores scalar metadata + a pointer to the per-run GeoPackage
written by this module.

This module is QGIS-only -- it uses ``QgsVectorFileWriter`` for the
write side and ``QgsVectorLayer(... 'ogr')`` for the load side.  The
OMRAT plugin invokes :func:`write_run_results` from the main thread
after :class:`compute.calculation_task.CalculationTask` finishes, and
:func:`load_run_results_to_map` when the user clicks "Add results to
map" on the Previous-runs table.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Layer-name suffixes used inside the per-run GeoPackage.  Keep these
# stable so older gpkg files keep loading.
_LAYER_NAMES = (
    'drifting_allision',
    'drifting_grounding',
    'powered_allision',
    'powered_grounding',
    'collision_lines',
    'collision_points',
)


def _build_memory_layers(
    calc: Any,
    structures: list[dict[str, Any]] | None,
    depths: list[dict[str, Any]] | None,
    segment_data: dict[str, Any] | None,
    depths_original: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the existing factory functions but with ``add_to_project=False``.

    Returns a dict of ``{layer_name: QgsVectorLayer | None}``.
    Empty / missing layers become ``None`` and are filtered out
    later.  Reuses the layer factories already in
    ``geometries.result_layers``.

    ``depths`` is what the **drifting** report's ``by_object`` keys
    reference -- the merged-meta list when threshold merging is on,
    the split list otherwise.

    ``depths_original`` is the per-original-data-depth list (with
    ids like ``"1"`` / ``"2"`` and a WGS84 geometry) needed by
    **powered grounding**, whose ``by_obstacle`` keys are the
    original data ids straight from ``data['depths']``.  Falls back
    to ``depths`` when the calc didn't provide it (older runs).
    """
    from geometries.result_layers import (
        create_result_layers,
        create_collision_layers,
        create_powered_grounding_layer,
        create_powered_allision_layer,
    )
    layers: dict[str, Any] = {n: None for n in _LAYER_NAMES}

    drifting_report = getattr(calc, 'drifting_report', None)
    if drifting_report is not None:
        try:
            allision_layer, grounding_layer = create_result_layers(
                drifting_report,
                structures or [], depths or [],
                add_to_project=False,
            )
            layers['drifting_allision'] = allision_layer
            layers['drifting_grounding'] = grounding_layer
        except Exception as exc:
            logger.warning(f"drifting result-layer build failed: {exc}")

    collision_report = getattr(calc, 'collision_report', None)
    if collision_report is not None and segment_data:
        try:
            line_layer, point_layer = create_collision_layers(
                collision_report, segment_data, add_to_project=False,
            )
            layers['collision_lines'] = line_layer
            layers['collision_points'] = point_layer
        except Exception as exc:
            logger.warning(f"collision result-layer build failed: {exc}")

    pg_report = getattr(calc, 'powered_grounding_report', None)
    if pg_report is not None:
        try:
            powered_depths = depths_original or depths or []
            layers['powered_grounding'] = create_powered_grounding_layer(
                pg_report, powered_depths, add_to_project=False,
            )
        except Exception as exc:
            logger.warning(f"powered-grounding layer build failed: {exc}")

    pa_report = getattr(calc, 'powered_allision_report', None)
    if pa_report is not None:
        try:
            layers['powered_allision'] = create_powered_allision_layer(
                pa_report, structures or [], add_to_project=False,
            )
        except Exception as exc:
            logger.warning(f"powered-allision layer build failed: {exc}")

    return layers


def write_run_results(
    calc: Any,
    output_path: str | Path,
    *,
    structures: list[dict[str, Any]] | None = None,
    depths: list[dict[str, Any]] | None = None,
    depths_original: list[dict[str, Any]] | None = None,
    segment_data: dict[str, Any] | None = None,
) -> list[str]:
    """Write every non-empty result layer to ``output_path`` as a
    GeoPackage.

    Returns the list of layer names that were written (subset of
    :data:`_LAYER_NAMES`).  Existing files are overwritten.

    ``depths`` is the drifting report's keying (merged-or-split, i.e.
    ``calc._last_depths``); ``depths_original`` is the per-original-id
    list used by powered grounding (``calc._last_depths_original``).
    """
    from qgis.core import (
        QgsVectorFileWriter,
        QgsCoordinateTransformContext,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        try:
            out.unlink()
        except Exception as exc:
            logger.warning(f"Couldn't remove existing {out}: {exc}")

    layers = _build_memory_layers(
        calc, structures, depths, segment_data,
        depths_original=depths_original,
    )
    written: list[str] = []
    transform_ctx = QgsCoordinateTransformContext()
    first_write = True

    for layer_name in _LAYER_NAMES:
        layer = layers.get(layer_name)
        if layer is None or layer.featureCount() == 0:
            continue
        opts = QgsVectorFileWriter.SaveVectorOptions()
        opts.driverName = 'GPKG'
        opts.layerName = layer_name
        # Resolve enums in a Qt5/Qt6-compatible way.
        opts.actionOnExistingFile = (
            QgsVectorFileWriter.CreateOrOverwriteFile if first_write
            else QgsVectorFileWriter.CreateOrOverwriteLayer
        )
        try:
            result = QgsVectorFileWriter.writeAsVectorFormatV3(
                layer, str(out), transform_ctx, opts,
            )
        except AttributeError:
            # Older QGIS without writeAsVectorFormatV3 -- fall back.
            result = QgsVectorFileWriter.writeAsVectorFormatV2(
                layer, str(out), transform_ctx, opts,
            )
        # Different QGIS versions return tuples of varying length;
        # the first element is the error code regardless.
        err = result[0] if isinstance(result, (tuple, list)) else result
        if err == QgsVectorFileWriter.NoError:
            written.append(layer_name)
            first_write = False
        else:
            err_msg = (
                result[1] if isinstance(result, (tuple, list)) and len(result) > 1
                else 'unknown'
            )
            logger.warning(
                f"Failed to write {layer_name} to {out}: code={err}, msg={err_msg}"
            )

    logger.info(f"Wrote {len(written)} layers to {out}")
    return written


# ---------------------------------------------------------------------------
# Loader (used by "Add results to map" button)
# ---------------------------------------------------------------------------

def _human_layer_name(layer_name: str, run_label: str) -> str:
    pretty = layer_name.replace('_', ' ').title()
    return f"{run_label} - {pretty}"


def _apply_default_styling(layer_name: str, qgis_layer) -> None:
    """Reuse ``geometries.result_layers`` graduated symbology.

    The styling helpers don't care whether the layer is in-memory or
    OGR-backed; they only inspect attributes and call
    ``layer.setRenderer``.
    """
    try:
        from geometries.result_layers import (
            apply_graduated_symbology,
            _line_symbol_factory, _marker_symbol_factory,
            _polygon_symbol_factory, _apply_graduated,
        )
    except Exception:
        return
    if layer_name in ('drifting_allision', 'drifting_grounding'):
        apply_graduated_symbology(qgis_layer)
    elif layer_name in ('powered_grounding', 'powered_allision'):
        _apply_graduated(qgis_layer, 'total_prob', _polygon_symbol_factory)
    elif layer_name == 'collision_lines':
        _apply_graduated(qgis_layer, 'combined', _line_symbol_factory)
    elif layer_name == 'collision_points':
        _apply_graduated(qgis_layer, 'combined', _marker_symbol_factory)


def load_run_results_to_map(
    gpkg_path: str | Path,
    run_label: str,
    *,
    add_to_project: bool = True,
) -> list[Any]:
    """Open every layer in the per-run GeoPackage and (optionally) add
    each one to the QGIS project.

    Returns the list of ``QgsVectorLayer`` objects created.
    """
    from qgis.core import QgsVectorLayer, QgsProject

    gpkg = Path(gpkg_path)
    if not gpkg.is_file():
        logger.warning(f"Per-run GeoPackage not found: {gpkg}")
        return []

    layers: list[Any] = []
    for layer_name in _LAYER_NAMES:
        uri = f"{gpkg}|layername={layer_name}"
        display = _human_layer_name(layer_name, run_label)
        layer = QgsVectorLayer(uri, display, 'ogr')
        if not layer.isValid() or layer.featureCount() == 0:
            continue
        _apply_default_styling(layer_name, layer)
        if add_to_project:
            QgsProject.instance().addMapLayer(layer)
        layers.append(layer)
    logger.info(f"Loaded {len(layers)} layers from {gpkg}")
    return layers
