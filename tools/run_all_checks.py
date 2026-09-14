"""Run every pre-release check in one go.

Order (each step is one of the existing ``tools/`` entry points, so the
individual scripts stay the canonical place for their configuration):

1. **pytest** -- the whole ``tests/`` folder with the QGIS conftest.  The
   OSGeo4W Qt6 environment (``apps/qgis`` + ``apps/Qt6/bin`` on ``PATH``,
   ``apps/qgis/python`` on ``PYTHONPATH``) is set up here, so plain
   ``C:/OSGeo4W/apps/Python312/python.exe`` is all you need to launch it.
   Runs with ``-p no:faulthandler`` to silence the harmless
   ``0xc0000139`` first-chance DLL probe noise from pyarrow.
2. **Qt6 check** -- ``tools/run_qt6_check.py`` (plugins.qgis.org's
   pyqgis4-checker, locally).
3. **flake8** -- ``tools/run_flake8.py`` on the shipped surface.
4. **bandit** -- ``tools/run_bandit.py``; gates on medium+ findings.
5. **clear pycache** -- ``tools/clear_pycache.py`` so QGIS does not load
   stale bytecode after all of the above imported everything.

All steps run even if an earlier one fails (pass ``--stop-on-fail`` to
change that); the summary at the end lists the outcome and duration of
each step.  Output of every step is streamed live.

Usage::

    C:/OSGeo4W/apps/Python312/python.exe tools/run_all_checks.py
    C:/OSGeo4W/apps/Python312/python.exe tools/run_all_checks.py --stop-on-fail
    C:/OSGeo4W/apps/Python312/python.exe tools/run_all_checks.py --skip tests --skip bandit
    C:/OSGeo4W/apps/Python312/python.exe tools/run_all_checks.py --slow      # include @slow tests
    C:/OSGeo4W/apps/Python312/python.exe tools/run_all_checks.py --pytest-args "-k powered -x"

Exit codes:
    0  - every step passed
    1  - at least one step failed (see summary)
    2  - could not set up the OSGeo4W Qt6 environment
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent

sys.path.insert(0, str(TOOLS_DIR))
from run_qt6_check import _check_env, _qt6_env  # noqa: E402  (sibling tool)


@dataclass
class Step:
    key: str
    title: str
    cmd: list[str]
    needs_qgis_env: bool = False
    # Return codes that count as success (clear_pycache returns 0 only).
    ok_codes: tuple[int, ...] = (0,)
    # Filled after the run.
    rc: int | None = None
    seconds: float = 0.0
    summary: str = ""
    skipped: bool = False
    tail: list[str] = field(default_factory=list)


def _build_steps(args: argparse.Namespace) -> list[Step]:
    py = sys.executable
    pytest_cmd = [py, "-m", "pytest", "-q", "-p", "no:faulthandler", "tests"]
    if args.slow:
        pytest_cmd += ["-m", "slow or not slow"]
    if args.pytest_args:
        pytest_cmd += shlex.split(args.pytest_args)
    return [
        Step("tests", "pytest (QGIS conftest)", pytest_cmd, needs_qgis_env=True),
        Step("qt6", "Qt6 / QGIS 4 check", [py, str(TOOLS_DIR / "run_qt6_check.py")]),
        Step("flake8", "flake8", [py, str(TOOLS_DIR / "run_flake8.py")]),
        Step("bandit", "bandit", [py, str(TOOLS_DIR / "run_bandit.py")]),
        Step("pycache", "clear __pycache__", [py, str(TOOLS_DIR / "clear_pycache.py")]),
    ]


def _pytest_summary(lines: list[str]) -> str:
    """Return pytest's final ``N passed, M failed in Xs`` line if present."""
    for ln in reversed(lines):
        stripped = ln.strip("= ").strip()
        if (" passed" in stripped or " failed" in stripped or " error" in stripped) and " in " in stripped:
            return stripped
    return ""


def _run_step(step: Step, env: dict[str, str] | None) -> None:
    banner = f" {step.title} "
    print("\n" + banner.center(72, "="), flush=True)
    print("$ " + " ".join(step.cmd), flush=True)
    start = time.perf_counter()
    proc = subprocess.Popen(  # nosec B603 - fixed argv built from our own tool paths
        step.cmd, cwd=str(REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        print(line, flush=True)
    proc.wait()
    step.rc = proc.returncode
    step.seconds = time.perf_counter() - start
    step.tail = [ln for ln in lines if ln.strip()][-3:]
    if step.key == "tests":
        step.summary = _pytest_summary(lines)
    if not step.summary and step.tail:
        step.summary = step.tail[-1]


def _print_summary(steps: list[Step]) -> bool:
    print("\n" + " Summary ".center(72, "="))
    all_ok = True
    for s in steps:
        if s.skipped:
            status = "SKIP"
        elif s.rc in s.ok_codes:
            status = "PASS"
        else:
            status = f"FAIL (rc={s.rc})"
            all_ok = False
        dur = f"{s.seconds:6.1f}s" if not s.skipped else "      -"
        print(f"  {status:<12} {dur}  {s.title}")
        if s.summary and not s.skipped:
            print(f"               {s.summary[:100]}")
    print("=" * 72)
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED -- see the step output above.")
    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stop-on-fail", action="store_true",
                        help="abort after the first failing step (pycache is still cleared)")
    parser.add_argument("--skip", action="append", default=[],
                        choices=["tests", "qt6", "flake8", "bandit", "pycache"],
                        help="skip a step; may be given more than once")
    parser.add_argument("--slow", action="store_true",
                        help="include tests marked @slow (pytest.ini deselects them by default)")
    parser.add_argument("--pytest-args", default="",
                        help='extra arguments for pytest, e.g. "-k powered -x"')
    args = parser.parse_args(argv)

    steps = _build_steps(args)
    for s in steps:
        s.skipped = s.key in args.skip

    qgis_env = None
    if any(s.needs_qgis_env and not s.skipped for s in steps):
        qgis_env = _qt6_env()
        if qgis_env is None:
            print("ERROR: could not find an OSGeo4W install with apps/Qt6. Set OSGEO4W_ROOT.",
                  file=sys.stderr)
            return 2
        problem = _check_env(qgis_env)
        if problem:
            print(f"ERROR: QGIS/Qt6 environment not usable: {problem}", file=sys.stderr)
            return 2

    aborted = False
    for s in steps:
        if s.skipped:
            continue
        if aborted and s.key != "pycache":
            s.skipped = True
            continue
        _run_step(s, qgis_env if s.needs_qgis_env else None)
        if args.stop_on_fail and s.rc not in s.ok_codes:
            print(f"\n--stop-on-fail: '{s.title}' failed, skipping the remaining checks.")
            aborted = True

    return 0 if _print_summary(steps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
