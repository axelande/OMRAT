"""Helpers for tests that need a genuinely read-only file.

The read-only tests clear the write bits with ``chmod`` and expect the
following write to fail.  That holds for a normal user on Windows, Linux
and macOS, but **not** for root: the kernel skips the permission check for
uid 0, so ``os.access(path, os.W_OK)`` stays true and ``open(path, 'w')``
succeeds.  The GitHub CI runs pytest as root inside the ``qgis/qgis``
Docker image, which made these tests fail there while passing locally.

``make_read_only`` therefore verifies that the permission actually took
effect and skips the test otherwise -- the behaviour under test simply
cannot be produced for that user.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def make_read_only(path: Path) -> None:
    """Clear the write bits on ``path``; skip the test if the OS still lets
    the current user write to it (root, or a filesystem that ignores mode
    bits)."""
    path.chmod(path.stat().st_mode & ~_WRITE_BITS)
    if os.access(str(path), os.W_OK):
        pytest.skip(
            "file permissions are not enforced for this user (root in the CI "
            "container) -- read-only behaviour cannot be exercised here"
        )


def restore_writable(path: Path) -> None:
    """Undo ``make_read_only`` so pytest can clean up ``tmp_path``."""
    if path.exists():
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
