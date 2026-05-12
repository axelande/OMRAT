"""Plugin-side handler for the junction registry and transition matrix.

Owns the live ``dict[junction_id, Junction]`` while the plugin is open,
mediates the validation pass triggered by the *Update all distributions*
button, and brokers the matrix-editor dialog so the rest of the plugin
doesn't have to know about ``geometries.junctions`` directly.

Kept QGIS-import-light at module load so the standalone tests can
exercise the handler under ``pytest --noconftest``; the only QGIS imports
live inside the methods that actually drive the UI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

from geometries.junctions import (
    Junction,
    apply_geometric_defaults,
    apply_ais_defaults,
    build_junctions,
    deserialize_junctions,
    refresh_junction_registry,
    serialize_junctions,
    transition_share,
    validate_junctions,
)

if TYPE_CHECKING:
    from omrat import OMRAT


class Junctions:
    """Live junction registry for the active OMRAT project.

    The handler is created in ``OMRAT.__init__`` and persisted across
    project reloads; ``load_from_dict`` rebuilds the registry from the
    ``junctions`` block of the loaded ``.omrat`` file (falling back to
    geometric defaults if the block is empty), and ``to_dict`` snapshots
    it for ``GatherData``.
    """

    def __init__(self, parent: "OMRAT | None" = None) -> None:
        self.p = parent
        self.registry: dict[str, Junction] = {}

    # ------------------------------------------------------------------
    # Construction / persistence
    # ------------------------------------------------------------------

    def rebuild_from_segments(
        self,
        segment_data: dict[str, Any] | None = None,
        *,
        prefer_user: bool = True,
    ) -> None:
        """Rebuild the registry from the current ``segment_data``.

        When ``prefer_user`` is true (default) any existing user-edited
        matrices that still reference live legs are preserved; otherwise
        the registry is regenerated from scratch.
        """
        sd = self._segment_data(segment_data)
        if prefer_user and self.registry:
            self.registry = refresh_junction_registry(self.registry, sd)
        else:
            self.registry = build_junctions(sd)
            apply_geometric_defaults(self.registry, sd)

    def load_from_dict(
        self,
        payload: dict[str, dict[str, Any]] | None,
        segment_data: dict[str, Any] | None = None,
    ) -> None:
        """Restore the registry from a serialised ``junctions`` block.

        After loading we always refresh against the current segment list
        so disappeared legs are pruned and brand-new junctions get a
        geometric default.  User-edited rows survive that refresh as
        long as their referenced legs still exist.
        """
        sd = self._segment_data(segment_data)
        if payload:
            self.registry = deserialize_junctions(payload)
        else:
            self.registry = {}
        self.registry = refresh_junction_registry(self.registry, sd)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Snapshot the registry for ``GatherData.get_all_for_save``."""
        return serialize_junctions(self.registry)

    # ------------------------------------------------------------------
    # Reads used by the compute pipeline
    # ------------------------------------------------------------------

    def get(self, junction_id: str) -> Junction | None:
        return self.registry.get(junction_id)

    def share(
        self,
        junction_pt: tuple[float, float],
        in_leg_id: str,
        out_leg_id: str,
        *,
        default: float = 1.0,
    ) -> float:
        """Convenience proxy for :func:`geometries.junctions.transition_share`."""
        return transition_share(
            self.registry, junction_pt, in_leg_id, out_leg_id, default=default,
        )

    def warnings(self, segment_data: dict[str, Any] | None = None) -> list:
        return validate_junctions(self.registry, self._segment_data(segment_data))

    # ------------------------------------------------------------------
    # Defaults pipeline
    # ------------------------------------------------------------------

    def apply_geometric_defaults(
        self,
        segment_data: dict[str, Any] | None = None,
        *,
        overwrite_user: bool = False,
    ) -> int:
        return apply_geometric_defaults(
            self.registry, self._segment_data(segment_data),
            overwrite_user=overwrite_user,
        )

    def apply_ais_counts(
        self,
        counts_by_junction: dict[str, dict[str, dict[str, float]]],
        segment_data: dict[str, Any] | None = None,
        *,
        overwrite_user: bool = False,
    ) -> int:
        """Replace each non-user matrix with one derived from AIS transition counts."""
        return apply_ais_defaults(
            self.registry,
            counts_by_junction,
            self._segment_data(segment_data),
            overwrite_user=overwrite_user,
        )

    # ------------------------------------------------------------------
    # User edits
    # ------------------------------------------------------------------

    def set_row(
        self,
        junction_id: str,
        in_leg_id: str,
        row: dict[str, float],
    ) -> bool:
        """Update one row of a junction's matrix and mark it user-edited.

        Rows are normalised to sum to 1.0 (so the UI does not need to
        force the user to type exact percentages).  Returns ``True`` if
        the junction exists, ``False`` otherwise.
        """
        j = self.registry.get(junction_id)
        if j is None:
            return False
        cleaned: dict[str, float] = {}
        total = 0.0
        for out_leg, frac in (row or {}).items():
            try:
                v = float(frac)
            except (TypeError, ValueError):
                continue
            if v < 0:
                v = 0.0
            cleaned[str(out_leg)] = v
            total += v
        if total > 0:
            cleaned = {k: v / total for k, v in cleaned.items()}
        j.transitions[str(in_leg_id)] = cleaned
        j.source = 'user'
        return True

    def set_matrix(
        self,
        junction_id: str,
        matrix: dict[str, dict[str, float]],
    ) -> bool:
        """Replace the entire matrix for a junction (one user edit)."""
        j = self.registry.get(junction_id)
        if j is None:
            return False
        for in_leg, row in (matrix or {}).items():
            self.set_row(junction_id, in_leg, row)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _segment_data(
        self, override: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if override is not None:
            return override
        if self.p is not None:
            return getattr(self.p, 'segment_data', {}) or {}
        return {}

    def __iter__(self) -> Iterable[Junction]:
        return iter(self.registry.values())

    def __len__(self) -> int:
        return len(self.registry)
