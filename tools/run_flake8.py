"""Run Flake8 on the files that actually ship to plugins.qgis.org.

Why a wrapper script instead of just calling Flake8 directly?

1. The repo contains plenty of code that's NOT shipped to QGIS users:
   ``tests/``, ``tools/`` (this file), ``examples/`` (admin scripts),
   ``docs/``, ``help/source/`` (Sphinx source), ``zip_build/``.
   Scanning those would create false positives unrelated to what a
   plugins.qgis.org reviewer would see.  The ``SHIPPED_TARGETS`` list
   below mirrors ``pb_tool.cfg`` ``[files]`` — keep them in sync when
   you change the packaging.

2. This wrapper prints a compact summary and forwards a meaningful exit
   code so CI can report the issue count without necessarily blocking
   the build (use ``continue-on-error: true`` in the workflow step while
   working down the backlog).

Exit codes:
    0  - no issues found
    1  - one or more style issues found
    2  - Flake8 not installed
"""
from __future__ import annotations

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


def _flake8_available() -> bool:
    try:
        import flake8  # noqa: F401
    except ImportError:
        return False
    return True


def _run_flake8() -> tuple[list[str], int]:
    """Run flake8 and return (output_lines, exit_code)."""
    cmd = [sys.executable, "-m", "flake8", *SHIPPED_TARGETS]
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT),
        capture_output=True, text=True,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return lines, proc.returncode


def main() -> int:
    if not _flake8_available():
        print(
            "ERROR: flake8 is not installed.\n"
            "Install it with:\n"
            "    pip install flake8\n"
            "or via the dev requirements:\n"
            "    pip install -r requirements_dev.txt",
            file=sys.stderr,
        )
        return 2

    lines, rc = _run_flake8()

    if not lines:
        print("Flake8 gate: PASS (no issues on the shipped surface).")
        return 0

    # Group by error code for a compact summary.
    from collections import Counter
    codes: Counter[str] = Counter()
    for ln in lines:
        # format: path:line:col: Exxx message
        parts = ln.split(":", 3)
        if len(parts) >= 4:
            code = parts[3].strip().split()[0]
            codes[code] += 1

    print(f"Flake8 found {len(lines)} issue(s) across the shipped targets.\n")
    print("Top codes:")
    for code, count in codes.most_common(10):
        print(f"  {code:>6}  {count}")

    print("\nFull output:")
    for ln in lines:
        print(" ", ln)

    print(
        f"\nFlake8 gate: {len(lines)} issue(s) found.\n"
        "Fix them or adjust .flake8 if a rule doesn't apply to this project."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
