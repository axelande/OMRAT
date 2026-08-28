"""Keep the ``file.py:NNN`` source references in the Sphinx docs honest.

The documentation cites the implementation in two places that both carry
a line number::

    ``compute/basic_equations.py:329`` -- `get_head_on_collision_candidates()
    <https://github.com/axelande/OMRAT/blob/main/compute/basic_equations.py#L329>`__

Nothing keeps those numbers in step with the code, so any refactor
silently points readers (and the GitHub deep-links) at the wrong lines.
Before the first run of this script, 58 of 73 references in the docs were
stale -- some by several hundred lines.

The symbol name is treated as authoritative: the script looks up where
``def``/``class <name>`` actually lives and rewrites both the inline
number and the ``#Lnnn`` anchor.

Usage
-----

Report without writing (use this in a pre-release check)::

    C:/OSGeo4W/apps/Python312/python.exe tools/check_doc_source_refs.py --check

Rewrite the stale numbers in place::

    C:/OSGeo4W/apps/Python312/python.exe tools/check_doc_source_refs.py

Exit status is 1 when stale or unresolvable references remain, so it can
gate a release.

Notes
-----

* References with no line number (whole-file citations) are checked only
  for the file existing.
* A symbol that no longer exists is reported for manual attention and
  left untouched -- it usually means the function moved to another
  module and the whole reference needs rewriting, not just the number.
* ``literalinclude`` blocks should use ``:pyobject:`` rather than
  ``:lines:``; that form cannot go stale and needs no maintenance.
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(REPO_ROOT, 'help', 'source')

# ``compute/basic_equations.py:329`` -- `symbol() <...#L329>`__
REF = re.compile(
    r'``(?P<path>[\w./_-]+\.py):(?P<line>\d+)``'
    r'(?P<mid>\s*--\s*`)(?P<sym>[A-Za-z_][\w.]*)(?P<call>\([^)]*\))?'
    r'(?P<gap>\s*<[^>]*?)#L(?P<urlline>\d+)(?P<tail>>`__)'
)

# Whole-file citations: ``path/to/file.py`` with no line number.  The
# ``/`` is required so bare module names in prose (``storage.py``) are
# not mistaken for path claims.
FILE_ONLY = re.compile(r'``(?P<path>[\w._-]+(?:/[\w._-]+)+\.py)``')

DEF = re.compile(r'^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)')

_symbol_cache: dict[str, dict[str, list[int]] | None] = {}
_source_cache: dict[str, list[str] | None] = {}

# How far from the cited line the identifier may appear and still count as
# "this reference points at the right place".  Covers a decorator or a
# wrapped signature above the line that was cited.
ANCHOR_TOLERANCE = 3


def symbol_lines(relpath: str) -> dict[str, list[int]] | None:
    """Map ``symbol -> [line numbers]`` for one module, or None if absent."""
    if relpath in _symbol_cache:
        return _symbol_cache[relpath]
    target = os.path.join(REPO_ROOT, relpath.replace('/', os.sep))
    try:
        source = io.open(target, encoding='utf-8').read()
    except OSError:
        _symbol_cache[relpath] = None
        return None
    table: dict[str, list[int]] = {}
    for lineno, line in enumerate(source.splitlines(), 1):
        match = DEF.match(line)
        if match:
            table.setdefault(match.group(1), []).append(lineno)
    _symbol_cache[relpath] = table
    return table


def source_lines(relpath: str) -> list[str] | None:
    """Raw lines of one module, or None if it is not on disk."""
    if relpath not in _source_cache:
        target = os.path.join(REPO_ROOT, relpath.replace('/', os.sep))
        try:
            _source_cache[relpath] = io.open(
                target, encoding='utf-8',
            ).read().splitlines()
        except OSError:
            _source_cache[relpath] = None
    return _source_cache[relpath]


def anchored_at(relpath: str, lineno: int, identifier: str) -> bool:
    """Is ``identifier`` present within a few lines of ``lineno``?

    Used for references that name something other than a function or
    class -- a dataclass field, a module constant, or a statement such as
    a signal connection.  Those cannot be relocated automatically, but
    they can still be *verified*: if the identifier is where the docs say
    it is, the reference is good.
    """
    lines = source_lines(relpath)
    if lines is None:
        return False
    word = re.compile(r'\b%s\b' % re.escape(identifier))
    lo = max(1, lineno - ANCHOR_TOLERANCE)
    hi = min(len(lines), lineno + ANCHOR_TOLERANCE)
    return any(word.search(lines[i - 1]) for i in range(lo, hi + 1))


def process(check_only: bool) -> int:
    rewritten = 0
    already_ok = 0
    manual: list[tuple[str, str, str, str, str]] = []
    missing_files: set[tuple[str, str]] = set()

    for rst in sorted(glob.glob(os.path.join(DOCS_DIR, '*.rst'))):
        text = io.open(rst, encoding='utf-8').read()
        name = os.path.basename(rst)

        def repl(m: re.Match) -> str:
            nonlocal rewritten, already_ok
            relpath = m.group('path')
            symbol = m.group('sym').split('.')[-1]
            table = symbol_lines(relpath)
            if table is None:
                manual.append(
                    (name, relpath, m.group('line'), symbol, 'file missing'),
                )
                return m.group(0)
            hits = table.get(symbol)
            if not hits:
                # Not a def/class.  It may still be a legitimate citation
                # of a field, constant or statement -- verify in place
                # rather than reporting a false positive.
                if anchored_at(relpath, int(m.group('line')), symbol):
                    already_ok += 1
                    return m.group(0)
                manual.append(
                    (name, relpath, m.group('line'), symbol,
                     'not found at or near the cited line'),
                )
                return m.group(0)
            old = int(m.group('line'))
            # Overloaded / repeated names: keep the closest definition so a
            # deliberate choice between them is not silently flipped.
            new = min(hits, key=lambda d: abs(d - old))
            if new == old and m.group('urlline') == m.group('line'):
                already_ok += 1
                return m.group(0)
            rewritten += 1
            return (
                '``%s:%d``%s%s%s%s#L%d%s' % (
                    relpath, new, m.group('mid'), m.group('sym'),
                    m.group('call') or '', m.group('gap'), new,
                    m.group('tail'),
                )
            )

        new_text = REF.sub(repl, text)

        # Whole-file citations: only the path is checkable.
        for m in FILE_ONLY.finditer(new_text):
            relpath = m.group('path')
            if symbol_lines(relpath) is None:
                missing_files.add((name, relpath))

        if new_text != text and not check_only:
            io.open(rst, 'w', encoding='utf-8', newline='\n').write(new_text)
            print('rewrote %s' % name)

    verb = 'stale' if check_only else 'rewritten'
    print()
    print('line references %-10s : %d' % (verb, rewritten))
    print('already correct            : %d' % already_ok)
    print('need manual attention      : %d' % len(manual))
    for row in manual:
        print('   %-24s %-38s :%-6s %-42s %s' % row)
    if missing_files:
        print('referenced files that do not exist : %d' % len(missing_files))
        for row in sorted(missing_files):
            print('   %-24s %s' % row)

    if manual or missing_files:
        return 1
    if check_only and rewritten:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--check', action='store_true',
        help='report stale references without rewriting them',
    )
    args = parser.parse_args()
    return process(args.check)


if __name__ == '__main__':
    sys.exit(main())
