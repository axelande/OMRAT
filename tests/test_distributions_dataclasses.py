"""Unit tests for the dataclasses in omrat_utils/handle_distributions.py.

The ``Distributions`` class itself wires up matplotlib and QGIS widgets
and is exercised by the full plugin tests (``test_load_data`` /
cascade integration).  This file focuses on the plain dataclasses
(``Normal``, ``Uniform``, ``Params``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.handle_distributions import Normal, Params, Uniform


class TestNormalDataclass:
    def test_default_instance(self):
        n = Normal()
        assert n.mean is None
        assert n.std == 0.0
        assert n.probability == 0.0

    def test_explicit_values(self):
        n = Normal(mean=5.0, std=2.0, probability=0.3)
        assert n.mean == 5.0 and n.std == 2.0 and n.probability == 0.3


class TestUniformDataclass:
    def test_defaults(self):
        u = Uniform()
        assert u.lower == 0.0
        assert u.upper == 0.0
        assert u.probability == 100.0

    def test_explicit_values(self):
        u = Uniform(lower=-5.0, upper=5.0, probability=0.5)
        assert u.lower == -5.0 and u.upper == 5.0


class TestParamsDataclass:
    def test_default_is_three_normals_plus_uniform(self):
        p = Params()
        components = list(p)
        assert len(components) == 4
        assert isinstance(components[0], Normal)
        assert isinstance(components[1], Normal)
        assert isinstance(components[2], Normal)
        assert isinstance(components[3], Uniform)

    def test_default_factory_creates_distinct_instances(self):
        """Each `Params()` call gets a fresh set of Normals/Uniform --
        regression guard against the mutable-default-arg pitfall."""
        p1 = Params()
        p2 = Params()
        p1.normal1.mean = 42.0
        assert p2.normal1.mean is None

    def test_iter_preserves_order(self):
        p = Params()
        order = list(p)
        assert order == [p.normal1, p.normal2, p.normal3, p.uniform]

    def test_populate_via_attribute_access(self):
        p = Params()
        p.normal1 = Normal(mean=1.0, std=0.5, probability=0.7)
        p.uniform = Uniform(lower=-1.0, upper=1.0, probability=0.3)
        assert list(p)[0].mean == 1.0
        assert list(p)[-1].lower == -1.0
