"""Run Bandit on the files that actually ship to plugins.qgis.org.

Why a wrapper script instead of just calling Bandit directly?

1. The repo contains plenty of code that's NOT shipped to QGIS users:
   ``tests/``, ``tools/`` (this file), ``examples/`` (admin scripts),
   ``docs/``, ``help/source/`` (Sphinx source), ``zip_build/``,
   ``plugin_upload.py`` (dev-only).  Scanning those would create false
   positives (e.g. f-string SQL in admin scripts, xmlrpc client in the
   upload tool) that have nothing to do with what a plugins.qgis.org
   reviewer would see.  The ``SHIPPED_TARGETS`` list below mirrors
   ``pb_tool.cfg`` ``[files]`` — keep them in sync when you change the
   packaging.

2. Bandit 1.9.x's process exit code does not reliably gate on findings
   that pass the severity / confidence filters (see upstream issue
   tracker).  This wrapper parses the JSON output and sets a definitive
   exit code itself, so CI gating is dependable.

Exit codes:
    0  - no medium+ findings (safe to publish)
    1  - one or more medium / high findings (must be fixed or marked
         ``# nosec BXXX -- <reason>``)
    2  - Bandit not installed
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Targets: mirrors ``pb_tool.cfg`` ``[files]`` (python_files + extra_dirs).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

SHIPPED_TARGETS: list[str] = [
    "__init__.py",
    "omrat.py",
    "omrat_widget.py",
    "omrat_utils",
    "compute",
    "geometries",
    "ui",
    "helpers",
    "drifting",
]

GATE_SEVERITIES = ("MEDIUM", "HIGH")


def _bandit_available() -> bool:
    try:
        import bandit  # noqa: F401
    except ImportError:
        return False
    return True


def _run_bandit_json() -> tuple[dict, int]:
    """Run bandit and return (parsed_json, raw_exit_code).

    Bandit prints JSON to stdout when ``--format json`` is set.  We don't
    use the raw exit code for gating — see module docstring.
    """
    cmd = [
        sys.executable, "-m", "bandit",
        "-r", *SHIPPED_TARGETS,
        "--format", "json",
        "--quiet",
    ]
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT),
        capture_output=True, text=True,
    )
    try:
        data = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except json.JSONDecodeError as exc:
        print(
            f"ERROR: could not parse bandit JSON output: {exc}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}",
            file=sys.stderr,
        )
        sys.exit(2)
    return data, proc.returncode


def _print_finding(idx: int, f: dict) -> None:
    sev = f.get("issue_severity", "?")
    conf = f.get("issue_confidence", "?")
    test_id = f.get("test_id", "?")
    name = f.get("test_name", "?")
    fname = f.get("filename", "?")
    line = f.get("line_number", "?")
    text = (f.get("issue_text") or "").strip()
    print(
        f"\n[{idx}] {sev}/{conf}  {test_id}  {name}\n"
        f"    at {fname}:{line}\n"
        f"    {text}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--all", action="store_true",
        help="Also list low-severity findings (informational; does not "
             "change the exit code).",
    )
    args = parser.parse_args(argv)

    if not _bandit_available():
        print(
            "ERROR: bandit is not installed.\n"
            "Install it with:\n"
            "    pip install bandit\n"
            "or via the dev requirements:\n"
            "    pip install -r requirements_dev.txt",
            file=sys.stderr,
        )
        return 2

    data, _ = _run_bandit_json()
    findings = data.get("results") or []

    blocking = [
        f for f in findings
        if str(f.get("issue_severity", "")).upper() in GATE_SEVERITIES
    ]
    informational = [
        f for f in findings
        if str(f.get("issue_severity", "")).upper() == "LOW"
    ]

    metrics = (data.get("metrics") or {}).get("_totals") or {}
    loc = metrics.get("loc", "?")
    nosec = metrics.get("nosec", "?")

    print(f"Bandit scanned {loc} lines across the shipped targets.")
    print(f"  Suppressed via # nosec: {nosec}")
    print(f"  Low-severity findings:  {len(informational)}")
    print(f"  Medium-severity:        "
          f"{sum(1 for f in blocking if f.get('issue_severity', '').upper() == 'MEDIUM')}")
    print(f"  High-severity:          "
          f"{sum(1 for f in blocking if f.get('issue_severity', '').upper() == 'HIGH')}")

    if blocking:
        print("\n" + "=" * 68)
        print(" GATE FAILED -- medium- or high-severity findings:")
        print("=" * 68)
        for idx, f in enumerate(blocking, 1):
            _print_finding(idx, f)
        print(
            "\nFix each finding, or add ``# nosec BXXX -- <reason>`` on\n"
            "the offending line if it's a false positive (the reason is\n"
            "required for audit).",
            file=sys.stderr,
        )
        return 1

    if args.all and informational:
        print("\n" + "=" * 68)
        print(" Low-severity findings (informational only)")
        print("=" * 68)
        for idx, f in enumerate(informational, 1):
            _print_finding(idx, f)

    print(
        "\nBandit gate: PASS (no medium+ findings on the shipped surface).\n"
        "Safe to package and upload to plugins.qgis.org."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
