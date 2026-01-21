# -*- coding: utf-8 -*-
"""
QgsTask wrapper for running drift corridor generation in a background thread.

This allows the QGIS UI to remain responsive during long calculations
and provides proper progress reporting through the QGIS task manager.
"""
from typing import Any, Optional
from qgis.core import QgsTask, QgsMessageLog, Qgis
from qgis.PyQt.QtCore import pyqtSignal


class DriftCorridorTask(QgsTask):
    """
    Background task for generating drift corridors.

    This task runs the corridor generation in a separate thread,
    allowing the QGIS UI to remain responsive.

    Signals:
        progress_updated: Emitted with (completed, total, message) when progress changes
        corridors_generated: Emitted with list of corridor dicts when complete
        generation_failed: Emitted with error message if generation fails
    """

    progress_updated = pyqtSignal(int, int, str)
    corridors_generated = pyqtSignal(list)
    generation_failed = pyqtSignal(str)

    def __init__(
        self,
        description: str,
        generator: Any,
        depth_threshold: float,
        height_threshold: float,
        target_prob: float = 1e-3,
    ):
        """
        Initialize the drift corridor task.

        Args:
            description: Human-readable description for the task
            generator: The DriftCorridorGenerator object
            depth_threshold: Depths <= this value create shadows
            height_threshold: Heights <= this value create shadows
            target_prob: Target probability for projection distance
        """
        super().__init__(description, QgsTask.CanCancel)
        self.generator = generator
        self.depth_threshold = depth_threshold
        self.height_threshold = height_threshold
        self.target_prob = target_prob
        self.exception: Optional[Exception] = None
        self.error_msg: Optional[str] = None
        self.corridors: list[dict] = []

    def run(self) -> bool:
        """
        Execute the corridor generation in a background thread.

        Returns:
            True if successful, False if failed or cancelled
        """
        QgsMessageLog.logMessage(
            'Starting drift corridor generation (v2)...',
            'OMRAT',
            Qgis.Info
        )

        try:
            def progress_callback(completed: int, total: int, message: str) -> bool:
                """Update progress and check for cancellation."""
                if self.isCanceled():
                    return False

                if total > 0:
                    progress_pct = int((completed / total) * 100)
                    self.setProgress(progress_pct)

                self.progress_updated.emit(completed, total, message)
                return True

            self.generator.set_progress_callback(progress_callback)

            self.corridors = self.generator.generate_corridors(
                self.depth_threshold,
                self.height_threshold,
                self.target_prob
            )

            if self.isCanceled():
                QgsMessageLog.logMessage(
                    'Drift corridor generation was cancelled by user',
                    'OMRAT',
                    Qgis.Warning
                )
                return False

            QgsMessageLog.logMessage(
                f'Drift corridor generation completed: {len(self.corridors)} corridors',
                'OMRAT',
                Qgis.Success
            )
            return True

        except Exception as e:
            self.exception = e
            self.error_msg = str(e)
            QgsMessageLog.logMessage(
                f'Drift corridor generation failed: {self.error_msg}',
                'OMRAT',
                Qgis.Critical
            )
            return False

    def finished(self, result: bool) -> None:
        """
        Called when the task completes (success or failure).

        This method runs in the main thread and can safely interact with Qt widgets.

        Args:
            result: True if successful, False if failed or cancelled
        """
        if result:
            self.corridors_generated.emit(self.corridors)
            QgsMessageLog.logMessage(
                'Drift corridor task finished successfully',
                'OMRAT',
                Qgis.Success
            )
        elif self.isCanceled():
            QgsMessageLog.logMessage(
                'Drift corridor task was cancelled',
                'OMRAT',
                Qgis.Warning
            )
        else:
            error_msg = self.error_msg or "Unknown error"
            self.generation_failed.emit(error_msg)
            QgsMessageLog.logMessage(
                f'Drift corridor task failed: {error_msg}',
                'OMRAT',
                Qgis.Critical
            )

    def cancel(self) -> None:
        """Called when the user cancels the task."""
        QgsMessageLog.logMessage(
            'Cancelling drift corridor task...',
            'OMRAT',
            Qgis.Warning
        )
        super().cancel()
