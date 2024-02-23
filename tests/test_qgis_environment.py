# coding=utf-8
"""Tests for QGIS functionality."""

import pytest
from qgis.core import (
    QgsProviderRegistry,
    QgsCoordinateReferenceSystem,
    )



def test_qgis_environment():
    """QGIS environment has the expected providers"""

    r = QgsProviderRegistry.instance()
    assert 'gdal' in r.providerList()
    assert 'ogr' in r.providerList()

def test_projection():
    """Test that QGIS properly parses a wkt string.
    """
    wkt = ('''GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.0174532925199433,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]''')
    crs = QgsCoordinateReferenceSystem()
    crs.createFromProj(wkt)
    auth_id = crs.authid()
    expected_auth_id = 'EPSG:4326'
    assert auth_id == expected_auth_id

