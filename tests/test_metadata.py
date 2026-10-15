"""metadata.txt must parse the way plugins.qgis.org and QGIS read it.

Both use ``configparser.ConfigParser`` with the default interpolation, so a
bare ``%`` anywhere in a value (e.g. "10 %" in the changelog) rejects the
upload: "'%' must be followed by '%' or '('" (v0.17.1, 2026-10-08).
"""
import configparser
import os

METADATA = os.path.join(os.path.dirname(__file__), '..', 'metadata.txt')


def test_metadata_parses_with_interpolation():
    cp = configparser.ConfigParser()
    cp.read(METADATA, encoding='utf-8')
    for key in cp['general']:
        cp.get('general', key)  # raises InterpolationSyntaxError on a bare '%'
    assert cp.get('general', 'version')
