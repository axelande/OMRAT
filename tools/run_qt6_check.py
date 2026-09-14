"""Run the plugins.qgis.org Qt6 compatibility check locally.

plugins.qgis.org runs ``pyqgis4-checker`` on every uploaded zip.  That
checker is a Docker wrapper around ``scripts/pyqt5_to_pyqt6/pyqt5_to_pyqt6.py``
from the QGIS repository, invoked as::

    pyqt5_to_pyqt6.py --dry_run --logfile pyqt6_checker.log .

The findings show up in the "Qt6 Check" tab of the plugin version page.
This wrapper reproduces that run on Windows with the OSGeo4W install:

1. Puts the Qt6 DLLs and the Qt6 build of QGIS on ``PATH`` /
   ``PYTHONPATH`` so ``import PyQt6`` and ``import qgis.core`` work.
   (Plain ``C:/OSGeo4W/apps/Python312/python.exe`` finds the PyQt6
   package but not ``Qt6Core.dll``.)
2. Runs the vendored, unmodified upstream ``tools/pyqt5_to_pyqt6.py`` in
   dry-run mode against the repo root.
3. Drops findings in directories that are not shipped in the plugin zip
   (``tests/``, ``tools/``, ``examples/``, ...) so the report matches what
   plugins.qgis.org sees.  ``SHIPPED_TARGETS`` mirrors ``run_flake8.py``
   and ``pb_tool.cfg`` -- keep them in sync.

Usage::

    C:/OSGeo4W/apps/Python312/python.exe tools/run_qt6_check.py
    C:/OSGeo4W/apps/Python312/python.exe tools/run_qt6_check.py --all
    C:/OSGeo4W/apps/Python312/python.exe tools/run_qt6_check.py --logfile out.log

Requires ``pip install tokenize-rt`` in the OSGeo4W Python.  Refresh the
vendored checker with ``--update`` when upstream changes.

Exit codes:
    0  - no Qt6 incompatibilities on the shipped surface
    1  - one or more findings
    2  - environment problem (OSGeo4W / PyQt6 / tokenize-rt missing)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKER = Path(__file__).resolve().parent / "pyqt5_to_pyqt6.py"
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/qgis/QGIS/master/"
    "scripts/pyqt5_to_pyqt6/pyqt5_to_pyqt6.py"
)

# Mirrors tools/run_flake8.py and pb_tool.cfg [files].
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


def _osgeo4w_root() -> Path | None:
    env = os.environ.get("OSGEO4W_ROOT")
    candidates = [Path(env)] if env else []
    candidates += [Path(sys.executable).resolve().parents[2], Path("C:/OSGeo4W")]
    for root in candidates:
        if (root / "apps" / "Qt6" / "bin").is_dir():
            return root
    return None


def _qt6_env() -> dict[str, str] | None:
    """Environment equivalent to ``o4w_env.bat`` + ``qt6_env.bat`` + ``python-qgis.bat``."""
    root = _osgeo4w_root()
    if root is None:
        return None
    env = os.environ.copy()
    qt6_bin = root / "apps" / "Qt6" / "bin"
    qgis_dir = root / "apps" / "qgis"          # the Qt6 build; qgis-ltr is Qt5
    path_parts = [str(qgis_dir / "bin"), str(qt6_bin), str(root / "bin")]
    # Drop any Qt5 bin dir so the wrong DLLs are never picked up first.
    path_parts += [p for p in env.get("PATH", "").split(os.pathsep)
                   if p and "qt5" not in p.lower()]
    env["PATH"] = os.pathsep.join(path_parts)
    env["QT_PLUGIN_PATH"] = os.pathsep.join(
        [str(qgis_dir / "qtplugins"), str(root / "apps" / "Qt6" / "plugins")])
    env["QGIS_PREFIX_PATH"] = qgis_dir.as_posix()
    env["PYTHONPATH"] = str(qgis_dir / "python")
    env["PYTHONHOME"] = str(root / "apps" / "Python312")
    env["PYTHONUTF8"] = "1"
    env["GDAL_DATA"] = str(root / "apps" / "gdal" / "share" / "gdal")
    env["PROJ_DATA"] = str(root / "share" / "proj")
    env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    return env


def _check_env(env: dict[str, str]) -> str | None:
    probe = (
        "import PyQt6.QtCore, tokenize_rt; "
        "from PyQt6.Qsci import QsciScintilla; "
        "import qgis.core; "
        "print(PyQt6.QtCore.QT_VERSION_STR, qgis.core.Qgis.QGIS_VERSION)"
    )
    proc = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip()
        return err.splitlines()[-1] if err else "unknown import error"
    qt_ver, qgis_ver = proc.stdout.split()[:2]
    print(f"Qt6 check environment: Qt {qt_ver}, QGIS {qgis_ver}")
    return None


def _update_checker() -> int:
    import urllib.request
    data = urllib.request.urlopen(UPSTREAM_URL).read()  # nosec B310 - fixed https URL
    CHECKER.write_bytes(data)
    print(f"Updated {CHECKER.relative_to(REPO_ROOT)} ({len(data)} bytes) from upstream.")
    return 0


def _is_shipped(rel: str) -> bool:
    for target in SHIPPED_TARGETS:
        if rel == target or rel.startswith(target + "/"):
            return True
    return False


def _finding_path(line: str) -> str | None:
    """Return the file path from a ``path:line:col - msg`` / ``path: msg`` log line."""
    body = line
    drive = ""
    # Windows paths start with ``X:``; skip that drive colon when splitting.
    if len(line) > 2 and line[1] == ":" and line[2] in "\\/":
        drive, body = line[:2], line[2:]
    head = body.split(":", 1)[0]
    if not head.strip() or head.startswith("==="):
        return None
    return drive + head.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--all", action="store_true",
                        help="report findings in every directory, not only the shipped ones")
    parser.add_argument("--logfile", default=None,
                        help="also write the filtered report here (default: print only)")
    parser.add_argument("--update", action="store_true",
                        help="download the newest pyqt5_to_pyqt6.py from the QGIS repo and exit")
    parser.add_argument("--qgis3-incompatible-changes", action="store_true",
                        help="forwarded to the checker (report Qt6-only fixes that break QGIS 3)")
    args = parser.parse_args(argv)

    if args.update:
        return _update_checker()

    env = _qt6_env()
    if env is None:
        print("ERROR: could not find an OSGeo4W install with apps/Qt6. Set OSGEO4W_ROOT.",
              file=sys.stderr)
        return 2
    problem = _check_env(env)
    if problem:
        print(f"ERROR: Qt6 environment not usable: {problem}\n"
              "Needs the OSGeo4W 'qgis' (Qt6) package, PyQt6, PyQt6-QScintilla and\n"
              "    C:/OSGeo4W/apps/Python312/python.exe -m pip install tokenize-rt",
              file=sys.stderr)
        return 2

    raw_log = REPO_ROOT / "pyqt6_checker.raw.log"
    checker_args = ["--dry_run", "--logfile", str(raw_log)]
    if args.qgis3_incompatible_changes:
        checker_args.append("--qgis3-incompatible-changes")
    checker_args.append(".")
    # OSGeo4W ships PyQt5 next to PyQt6.  The checker probes ``import PyQt5``
    # and logs a warning if it succeeds; that call configures the root logger
    # before ``--logfile`` gets a chance, so the log file is never written.
    # The pyqgis4-checker Docker image has no PyQt5, so mimic that by making
    # the import fail inside the checker's interpreter.
    bootstrap = (
        "import runpy, sys; "
        "sys.modules['PyQt5'] = None; "
        f"sys.argv = [{str(CHECKER)!r}] + {checker_args!r}; "
        f"runpy.run_path({str(CHECKER)!r}, run_name='__main__')"
    )
    proc = subprocess.run([sys.executable, "-c", bootstrap], cwd=str(REPO_ROOT), env=env,
                          capture_output=True, text=True)
    if not raw_log.exists():
        print("ERROR: checker produced no log file.\n" + proc.stderr, file=sys.stderr)
        return 2

    findings: list[str] = []
    for line in raw_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("==="):
            continue
        if line.startswith("WARNING: PyQt5 has been found"):
            continue        # OSGeo4W ships both bindings; harmless for a dry run
        path = _finding_path(line)
        if path is None:
            continue
        rel = os.path.relpath(path, REPO_ROOT) if os.path.isabs(path) else path
        rel = rel.replace("\\", "/")
        if rel.startswith("./"):
            rel = rel[2:]
        if not args.all and not _is_shipped(rel):
            continue
        findings.append(rel + line[len(path):])
    raw_log.unlink(missing_ok=True)

    if args.logfile:
        text = "\n".join(findings) + ("\n" if findings else "")
        Path(args.logfile).write_text(text, encoding="utf-8")

    scope = "the whole repo" if args.all else "the shipped surface"
    if not findings:
        print(f"Qt6 check: PASS (no findings on {scope}).")
        return 0

    by_file: dict[str, int] = {}
    for ln in findings:
        key = ln.split(":", 1)[0]
        by_file[key] = by_file.get(key, 0) + 1
    print(f"Qt6 check found {len(findings)} finding(s) in {len(by_file)} file(s) on {scope}.\n")
    print("Per file:")
    for f, n in sorted(by_file.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {f}")
    print("\nFull output:")
    for ln in findings:
        print(" ", ln)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
