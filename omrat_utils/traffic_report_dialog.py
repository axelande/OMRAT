"""Traffic report: gather the project, save it as HTML and open it (Qt side).

**Traffic report...** on the Traffic Data tab and in the File menu call
:func:`open_traffic_report`; the messages after Update AIS / Update all
distributions, Copy traffic and Suppress leg carry a *Traffic report*
button (:func:`notify_with_report`).  The page itself is built by the pure
modules :mod:`omrat_utils.traffic_summary` (numbers and checks) and
:mod:`omrat_utils.traffic_report_html` (the HTML).

The report shows the live model (``GatherData.get_all_for_save``), unsaved
changes included.  The file name carries the time, so a report from before
and one from after a copy do not overwrite each other.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Optional

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtCore import QLocale, QSettings, QUrl
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton

from omrat_utils.traffic_report_html import render_traffic_report_html
from omrat_utils.traffic_summary import build_traffic_summary

LAST_DIR_KEY = 'omrat/traffic_report_dir'


def report_stem(omrat: Any) -> str:
    """File-name stem: the project file's, else the project name, else ``omrat``."""
    path = getattr(omrat, 'project_path', None)
    if path:
        return os.path.splitext(os.path.basename(path))[0]
    name = ''
    try:
        name = (omrat.main_widget.LEProjectName.text() or '').strip()
    except Exception:  # nosec B110 B112
        pass
    name = re.sub(r'[^\w\-]+', '_', name).strip('_')
    return name or 'omrat'


def _start_dir(omrat: Any) -> str:
    path = getattr(omrat, 'project_path', None)
    if path and os.path.isdir(os.path.dirname(path)):
        return os.path.dirname(path)
    last = QSettings().value(LAST_DIR_KEY, '', type=str)
    if last and os.path.isdir(last):
        return last
    try:
        out = omrat._get_output_dir()
    except Exception:  # nosec B110 B112
        out = None
    return str(out) if out else os.path.expanduser('~')


def default_report_path(omrat: Any, now: Optional[datetime] = None) -> str:
    """``<project dir>/<stem>_traffic_report_<YYYYMMDD_HHMM>.html``."""
    now = now or datetime.now()
    return os.path.join(_start_dir(omrat), f"{report_stem(omrat)}_traffic_report_{now:%Y%m%d_%H%M}.html")


def _decimal_point() -> str:
    try:
        return str(QLocale().decimalPoint()) or '.'
    except Exception:  # nosec B110 B112
        return '.'


def write_traffic_report(omrat: Any, path: str, data: Optional[dict[str, Any]] = None) -> str:
    """Build the report from the live model (or ``data``) and write it to
    ``path``.  Returns the path.  Headless core of :func:`open_traffic_report`."""
    if data is None:
        from omrat_utils.gather_data import GatherData
        data = GatherData(omrat).get_all_for_save()
    summary = build_traffic_summary(data, project_path=getattr(omrat, 'project_path', None) or '')
    page = render_traffic_report_html(summary, decimal=_decimal_point(), csv_stem=report_stem(omrat))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(page)
    return path


def open_traffic_report(omrat: Any, path: Optional[str] = None) -> Optional[str]:
    """Ask where to save the report (unless ``path`` is given), write it and
    open it in the browser.  Returns the path, or ``None`` when cancelled."""
    parent = getattr(omrat, 'main_widget', None)
    if not getattr(omrat, 'segment_data', None):
        QMessageBox.information(parent, omrat.tr("Traffic report"),
                                omrat.tr("There are no legs yet: draw or load a route first."))
        return None
    if path is None:
        path, _filter = QFileDialog.getSaveFileName(
            parent, omrat.tr("Save traffic report"), default_report_path(omrat),
            omrat.tr("HTML page (*.html)"))
        if not path:
            return None
        if not path.lower().endswith(('.html', '.htm')):
            path += '.html'
    try:
        write_traffic_report(omrat, path)
    except Exception as exc:
        QgsMessageLog.logMessage(f"Traffic report failed: {exc}", 'OMRAT', Qgis.MessageLevel.Critical)
        QMessageBox.warning(parent, omrat.tr("Traffic report"),
                            omrat.tr("The traffic report could not be written:\n\n{err}").format(err=exc))
        return None
    QSettings().setValue(LAST_DIR_KEY, os.path.dirname(path))
    if not getattr(omrat, 'testing', False):
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))
    notifier = getattr(omrat, 'notifier', None)
    if notifier is not None:
        try:
            notifier.display_message(omrat.tr("Traffic report saved to {path}").format(path=path), duration=8)
        except Exception:  # nosec B110 B112
            pass
    return path


def report_button(omrat: Any) -> Optional[QPushButton]:
    """A *Traffic report* button for a message-bar item (``None`` without a
    running Qt application, e.g. in a test with a mocked model)."""
    if QApplication.instance() is None:
        return None
    btn = QPushButton(omrat.tr("Traffic report"))
    btn.setToolTip(omrat.tr("Save and open an HTML report of the traffic on every leg."))
    btn.clicked.connect(lambda _checked=False: open_traffic_report(omrat))
    return btn


def notify_with_report(omrat: Any, text: str, duration: int = 10) -> None:
    """Message-bar message with a *Traffic report* button; a plain message
    when the button cannot be added."""
    notifier = getattr(omrat, 'notifier', None)
    if notifier is None:
        return
    try:
        btn = report_button(omrat)
        if btn is not None:
            notifier.display_message(text, duration=duration, widgets=[btn])
            return
    except Exception:  # nosec B110 B112
        pass
    try:
        notifier.display_message(text, duration=duration)
    except Exception:  # nosec B110 B112
        pass
