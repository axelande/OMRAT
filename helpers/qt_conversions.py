"""
Qt5/Qt6 Compatibility Layer for OMRAT

This module provides a compatibility layer that works with both Qt5 and Qt6
through QGIS's PyQt abstraction. It handles differences between Qt5 and Qt6
APIs to ensure the plugin works regardless of which Qt version QGIS is using.

Usage:
    from helpers.qt_conversions import QtCompat

    # Use QtCompat methods for version-specific operations
    enum_value = QtCompat.to_int(some_enum)
"""

from typing import Any
from qgis.PyQt import QtCore


class QtCompat:
    """
    Compatibility layer for Qt5/Qt6 differences.

    This class provides static methods to handle API differences between
    Qt5 and Qt6, allowing the same code to work with both versions.
    """

    # Detect Qt version
    QT_VERSION = QtCore.QT_VERSION_STR
    QT_MAJOR_VERSION = int(QT_VERSION.split('.')[0])
    IS_QT6 = QT_MAJOR_VERSION >= 6

    @staticmethod
    def to_int(enum_value: Any) -> int:
        """
        Convert enum to integer value.

        In Qt5, enums can be used directly as integers.
        In Qt6, enums are proper Python enums and need .value property.

        Args:
            enum_value: An enum value from Qt (e.g., QMetaType.QString)

        Returns:
            The integer value of the enum

        Example:
            >>> from qgis.PyQt.QtCore import QMetaType
            >>> QtCompat.to_int(QMetaType.QString)
        """
        if QtCompat.IS_QT6:
            # Qt6 uses proper enums that have a .value attribute
            return int(enum_value.value) if hasattr(enum_value, 'value') else int(enum_value)
        else:
            # Qt5 enums are already integers
            return int(enum_value)

    @staticmethod
    def from_qstring(qstring: Any) -> str:
        """
        Convert QString to Python string.

        In Qt5, QString exists and needs conversion.
        In Qt6, QString is removed and Python strings are used directly.

        Args:
            qstring: A QString (Qt5) or str (Qt6)

        Returns:
            Python string
        """
        if QtCompat.IS_QT6:
            # Qt6 uses Python strings directly
            return str(qstring)
        else:
            # Qt5 might have QString objects
            return str(qstring) if qstring is not None else ""

    @staticmethod
    def to_qbytearray(data: bytes | str) -> QtCore.QByteArray:
        """
        Create QByteArray from bytes or string.

        Handles differences in QByteArray creation between Qt versions.

        Args:
            data: Bytes or string to convert

        Returns:
            QByteArray instance
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        return QtCore.QByteArray(data)

    @staticmethod
    def get_variant_type(type_name: str) -> int:
        """
        Get QVariant type for a given type name.

        Qt5 uses QVariant.Type enum.
        Qt6 uses QMetaType enum (QVariant.Type is deprecated).

        Args:
            type_name: Name of the type (e.g., 'String', 'Int', 'Double')

        Returns:
            The type constant as integer
        """
        if QtCompat.IS_QT6:
            # Qt6: Use QMetaType
            from qgis.PyQt.QtCore import QMetaType
            type_map = {
                'String': QMetaType.Type.QString,
                'Int': QMetaType.Type.Int,
                'Double': QMetaType.Type.Double,
                'Bool': QMetaType.Type.Bool,
                'LongLong': QMetaType.Type.LongLong,
            }
            return QtCompat.to_int(type_map.get(type_name, QMetaType.Type.QString))
        else:
            # Qt5: Use QVariant.Type
            from qgis.PyQt.QtCore import QVariant
            type_map = {
                'String': QVariant.String,
                'Int': QVariant.Int,
                'Double': QVariant.Double,
                'Bool': QVariant.Bool,
                'LongLong': QVariant.LongLong,
            }
            return type_map.get(type_name, QVariant.String)

    @staticmethod
    def exec_dialog(dialog: Any) -> int:
        """
        Execute a dialog in a Qt version-compatible way.

        Qt5 uses exec_()
        Qt6 uses exec()

        Args:
            dialog: QDialog instance

        Returns:
            Dialog result code
        """
        if QtCompat.IS_QT6:
            return dialog.exec()
        else:
            return dialog.exec_()

    @staticmethod
    def sort_items(item_list: list, key: Any = None, reverse: bool = False) -> list:
        """
        Sort items in a Qt version-compatible way.

        Some Qt sorting methods changed between versions.

        Args:
            item_list: List to sort
            key: Sort key function
            reverse: Whether to reverse sort

        Returns:
            Sorted list
        """
        return sorted(item_list, key=key, reverse=reverse)

    @staticmethod
    def get_save_file_name(parent: Any, caption: str, directory: str, filter: str) -> tuple:
        """
        Get save file name in a Qt version-compatible way.

        Qt5 returns (filename, selected_filter)
        Qt6 may have slightly different return behavior

        Args:
            parent: Parent widget
            caption: Dialog caption
            directory: Initial directory
            filter: File filter string

        Returns:
            Tuple of (filename, selected_filter)
        """
        from qgis.PyQt.QtWidgets import QFileDialog
        result = QFileDialog.getSaveFileName(parent, caption, directory, filter)
        # Both Qt5 and Qt6 return tuple (filename, selected_filter)
        return result

    @staticmethod
    def get_open_file_name(parent: Any, caption: str, directory: str, filter: str) -> tuple:
        """
        Get open file name in a Qt version-compatible way.

        Args:
            parent: Parent widget
            caption: Dialog caption
            directory: Initial directory
            filter: File filter string

        Returns:
            Tuple of (filename, selected_filter)
        """
        from qgis.PyQt.QtWidgets import QFileDialog
        result = QFileDialog.getOpenFileName(parent, caption, directory, filter)
        return result


def get_qt_version() -> str:
    """
    Get the Qt version string.

    Returns:
        Qt version string (e.g., "5.15.2" or "6.2.0")
    """
    return QtCompat.QT_VERSION


def is_qt6() -> bool:
    """
    Check if running Qt6.

    Returns:
        True if Qt6, False if Qt5
    """
    return QtCompat.IS_QT6


def is_qt5() -> bool:
    """
    Check if running Qt5.

    Returns:
        True if Qt5, False if Qt6
    """
    return not QtCompat.IS_QT6
