"""User-adjustable parameters for AIS → linestring-segment ingestion.

These settings drive the TDKC compression that turns raw AIS pings into the
constant-COG/SOG linestring rows OMRAT stores in PostGIS.  Defaults are tuned
for maritime-risk-analysis use:

- ``min_sed_m = 30 m`` — just under typical AIS Class-A position accuracy
  (~10 m) plus a margin for the smallest vessel sizes OMRAT cares about.
  Sub-30 m position deviations are below the granularity of risk modelling
  and are filtered as noise.
- ``min_svd_kn = 0.3 kn`` — above realistic AIS speed jitter, well below
  any risk-relevant manoeuvre.

Both are higher than ``aissegments``' library defaults (``1.0 m`` / ``0.01 kn``)
which only filter floating-point noise.

Track-splitter knobs (applied *before* TDKC, so impossible jumps don't
become compressed segments):

- ``max_gap_s = 3600 s`` — split the track wherever the inter-ping gap
  exceeds one hour.  Ships routinely drop out of AIS coverage and reappear
  far away; bridging the two with a single TDKC segment produces the
  long, bogus straight lines that the user reported.
- ``speed_tolerance = 0.3`` — split when the implied speed (haversine /
  dt) exceeds the average reported SOG of the two pings by more than
  30%.  A relative tolerance instead of an absolute knot cap so it adapts
  to fast vessels (40 kn ferries are normal; 40 kn implied speed for a
  trawler reporting 8 kn is not).
- ``speed_floor_kn = 1.0`` — additive slack added to the limit so that
  GPS jitter near zero SOG (moored vessels reporting 0.0 kn but jiggling
  ~0.3 kn from satellite noise) does not trigger a split.

The wizard UI surfaces these as an editable form for each ingestion run; the
last-saved values persist per connection profile via QSettings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# Lazy Qt import: this module is also useful headless (CLI ingestion).
try:  # pragma: no cover - exercised only without Qt
    from qgis.PyQt.QtCore import QSettings
    _HAS_QT = True
except Exception:
    QSettings = None  # type: ignore[assignment]
    _HAS_QT = False


DEFAULT_MIN_SED_M = 30.0
DEFAULT_MIN_SVD_KN = 0.3
DEFAULT_MAX_GAP_S = 3600.0
DEFAULT_SPEED_TOLERANCE = 0.3
DEFAULT_SPEED_FLOOR_KN = 1.0
DEFAULT_PROFILE = "default"

_PROFILE_PREFIX = "omrat/ingest_profiles"


@dataclass
class IngestionSettings:
    """TDKC compression thresholds + plausibility splitter for one ingestion run.

    Use :meth:`from_qsettings` to load the per-profile defaults a user has
    saved, edit the fields in the wizard, then :meth:`to_qsettings` to
    persist the new values for next time.

    Attributes
    ----------
    name : str
        Profile name; matches the corresponding ``ConnectionProfile.name`` so
        each DB connection can have its own ingestion defaults.
    min_sed_m : float
        Lower bound (metres) for the adaptive SED threshold passed to
        ``aissegments.tdkc(min_sed_m=…)``.
    min_svd_kn : float
        Lower bound (knots) for the adaptive SVD threshold passed to
        ``aissegments.tdkc(min_svd_kn=…)``.
    max_gap_s : float
        Maximum allowed time gap between consecutive pings (seconds).  A
        larger gap forces a track split — the two sides become independent
        tracks for TDKC.
    speed_tolerance : float
        Allowed fractional excess of the *implied* speed (haversine
        distance / dt) over the average reported SOG of the two pings.
        ``0.3`` means split when implied > avg_sog × 1.3 + ``speed_floor_kn``.
    speed_floor_kn : float
        Additive slack (knots) on top of the tolerance limit, so GPS
        jitter for moored / slow vessels does not trigger spurious splits.
    """

    name: str = DEFAULT_PROFILE
    min_sed_m: float = DEFAULT_MIN_SED_M
    min_svd_kn: float = DEFAULT_MIN_SVD_KN
    max_gap_s: float = DEFAULT_MAX_GAP_S
    speed_tolerance: float = DEFAULT_SPEED_TOLERANCE
    speed_floor_kn: float = DEFAULT_SPEED_FLOOR_KN

    # ---------------------------------------------------------------- helpers

    def to_aissegments_kwargs(self) -> dict[str, float]:
        """Keyword arguments to splat into ``aissegments.tdkc(...)``."""
        return {"min_sed_m": float(self.min_sed_m), "min_svd_kn": float(self.min_svd_kn)}

    def to_splitter_kwargs(self) -> dict[str, float]:
        """Keyword arguments for the track-splitter (see :func:`split_track_at_gaps`)."""
        return {
            "max_gap_s": float(self.max_gap_s),
            "speed_tolerance": float(self.speed_tolerance),
            "speed_floor_kn": float(self.speed_floor_kn),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IngestionSettings":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    # ------------------------------------------------------------ persistence

    @classmethod
    def from_qsettings(cls, name: str = DEFAULT_PROFILE) -> "IngestionSettings":
        if not _HAS_QT:
            return cls(name=name)
        s = QSettings()
        prefix = f"{_PROFILE_PREFIX}/{name}"
        return cls(
            name=name,
            min_sed_m=float(
                s.value(f"{prefix}/min_sed_m", DEFAULT_MIN_SED_M, type=float)
                or DEFAULT_MIN_SED_M
            ),
            min_svd_kn=float(
                s.value(f"{prefix}/min_svd_kn", DEFAULT_MIN_SVD_KN, type=float)
                or DEFAULT_MIN_SVD_KN
            ),
            max_gap_s=float(
                s.value(f"{prefix}/max_gap_s", DEFAULT_MAX_GAP_S, type=float)
                or DEFAULT_MAX_GAP_S
            ),
            speed_tolerance=float(
                s.value(f"{prefix}/speed_tolerance", DEFAULT_SPEED_TOLERANCE, type=float)
                or DEFAULT_SPEED_TOLERANCE
            ),
            speed_floor_kn=float(
                s.value(f"{prefix}/speed_floor_kn", DEFAULT_SPEED_FLOOR_KN, type=float)
                or DEFAULT_SPEED_FLOOR_KN
            ),
        )

    def to_qsettings(self) -> None:
        if not _HAS_QT:
            return
        s = QSettings()
        prefix = f"{_PROFILE_PREFIX}/{self.name}"
        s.setValue(f"{prefix}/min_sed_m", float(self.min_sed_m))
        s.setValue(f"{prefix}/min_svd_kn", float(self.min_svd_kn))
        s.setValue(f"{prefix}/max_gap_s", float(self.max_gap_s))
        s.setValue(f"{prefix}/speed_tolerance", float(self.speed_tolerance))
        s.setValue(f"{prefix}/speed_floor_kn", float(self.speed_floor_kn))
