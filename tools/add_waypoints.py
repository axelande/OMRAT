"""Add the ``waypoints`` block (and ``start_wp`` / ``end_wp`` on every
leg) to ``.omrat`` files written before v0.15.2.

OMRAT derives the block itself whenever it loads a file, so this is not
required to open an old project.  Run it when you want the nodes to be
explicit in the file -- for version control diffs, for hand edits, or
before scripting against the file -- or to check a file's consistency.

Usage::

    python tools/add_waypoints.py project.omrat            # print report only
    python tools/add_waypoints.py project.omrat --write    # rewrite in place
    python tools/add_waypoints.py a.omrat b.omrat --write  # several files
    python tools/add_waypoints.py project.omrat --check    # exit 1 if inconsistent

Coincident leg endpoints (within about 1 cm) become one node.  Existing
ids in a file that already has the block are kept where the coordinates
still match.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometries.waypoints import (  # noqa: E402
    rebuild_waypoints, validate_waypoint_refs, waypoints_from_serializable, waypoints_to_serializable,
)


def process(path: Path, write: bool) -> tuple[int, int, list[str]]:
    """Returns ``(n_nodes, n_legs, problems)``; rewrites the file when asked."""
    data = json.loads(path.read_text(encoding='utf-8'))
    segment_data = data.get('segment_data') or {}
    before = waypoints_from_serializable(data.get('waypoints'))
    wps = rebuild_waypoints(segment_data, before)
    problems = validate_waypoint_refs(segment_data, wps)
    if write:
        data['waypoints'] = waypoints_to_serializable(wps)
        path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return len(wps), len(segment_data), problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+', type=Path)
    ap.add_argument('--write', action='store_true', help='rewrite each file in place')
    ap.add_argument('--check', action='store_true', help='exit 1 when any file is inconsistent')
    args = ap.parse_args(argv)
    bad = 0
    for path in args.files:
        if not path.is_file():
            print(f'{path}: not a file')
            bad += 1
            continue
        n_wp, n_legs, problems = process(path, args.write)
        state = 'written' if args.write else 'checked'
        print(f'{path}: {n_legs} legs, {n_wp} waypoints ({state})')
        for line in problems:
            print(f'  - {line}')
        if problems:
            bad += 1
    return 1 if (args.check and bad) else 0


if __name__ == '__main__':
    sys.exit(main())
