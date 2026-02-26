"""
OMRAT Helper Modules

This package contains helper utilities for the OMRAT plugin.
"""

from .qt_conversions import QtCompat, get_qt_version, is_qt6, is_qt5

__all__ = ['QtCompat', 'get_qt_version', 'is_qt6', 'is_qt5']
