"""
QgsTask wrapper for running probability calculations in a background thread.

This allows the QGIS UI to remain responsive during long calculations
and provides proper progress reporting through the QGIS task manager.
"""
from typing import Any, Callable, Optional
from qgis.core import QgsTask, QgsMessageLog, Qgis
from qgis.PyQt.QtCore import pyqtSignal


class CalculationTask(QgsTask):
    """
    Background task for running drifting model calculations.

    This task runs the expensive probability hole calculations in a separate thread,
    allowing the QGIS UI to remain responsive. Progress updates are sent via signals
    and displayed in the QGIS task manager.

    Signals:
        progress_updated: Emitted with (completed, total, message) when progress changes
        calculation_finished: Emitted when calculation completes successfully
        calculation_failed: Emitted with error message if calculation fails
    """

    # Custom signals for progress and completion
    progress_updated = pyqtSignal(int, int, str)  # (completed, total, message)
    calculation_finished = pyqtSignal(object)  # (calc object with results)
    calculation_failed = pyqtSignal(str)  # (error message)

    def __init__(
        self,
        description: str,
        calc_object: Any,
        data: dict[str, Any],
    ):
        """
        Initialize the calculation task.

        Args:
            description: Human-readable description for the task
            calc_object: The Calculation object with run_drifting_model method
            data: Dictionary of data passed to run_drifting_model
        """
        super().__init__(description, QgsTask.CanCancel)
        self.calc = calc_object
        self.data = data
        self.exception: Optional[Exception] = None
        self.error_msg: Optional[str] = None

        # Store reference to progress callback
        self._progress_callback: Optional[Callable[[int, int, str], None]] = None

    def set_progress_callback(self, callback: Callable[[int, int, str], None]) -> None:
        """
        Set a callback function for progress updates.

        Args:
            callback: Function that takes (completed, total, message)
        """
        self._progress_callback = callback

    def _update_description(self, text: str) -> None:
        """Update the task description shown in the QGIS task manager."""
        self.setDescription(f"OMRAT: {text}")

    def run(self) -> bool:
        """
        Execute the calculation in a background thread.

        This method is called by QgsTask when the task starts.
        It should not directly interact with Qt widgets.

        Returns:
            True if successful, False if failed or cancelled
        """
        QgsMessageLog.logMessage(
            'Starting calculation...',
            'OMRAT',
            Qgis.Info
        )

        try:
            # Inject progress callback into the calculation
            if hasattr(self.calc, 'set_progress_callback'):
                def progress_wrapper(completed: int, total: int, message: str) -> bool:
                    """Wrapper that updates task progress and checks for cancellation."""
                    if self.isCanceled():
                        return False  # Signal to stop calculation

                    # Update QgsTask progress (0-100)
                    if total > 0:
                        progress_pct = int((completed / total) * 100)
                        self.setProgress(progress_pct)

                    # Update task description with current status
                    self._update_description(message)

                    # Emit custom signal for detailed progress
                    self.progress_updated.emit(completed, total, message)

                    return True  # Continue calculation

                self.calc.set_progress_callback(progress_wrapper)

            # Phase 1: Drifting model
            self._update_description("Drifting model - preparing...")
            self.calc.run_drifting_model(self.data)

            if self.isCanceled():
                return False

            # Phase 2: Ship-ship collisions
            self._update_description("Ship-ship collisions...")
            self.setProgress(0)
            self.calc.run_ship_collision_model(self.data)

            if self.isCanceled():
                return False

            # Phase 3: Powered grounding
            self._update_description("Powered grounding...")
            self.setProgress(0)
            self.calc.run_powered_grounding_model(self.data)

            if self.isCanceled():
                return False

            # Phase 4: Powered allision
            self._update_description("Powered allision...")
            self.setProgress(0)
            self.calc.run_powered_allision_model(self.data)

            # Check if cancelled during execution
            if self.isCanceled():
                QgsMessageLog.logMessage(
                    'Calculation was cancelled by user',
                    'OMRAT',
                    Qgis.Warning
                )
                return False

            self._update_description("Complete")
            self.setProgress(100)
            QgsMessageLog.logMessage(
                'All calculations completed successfully',
                'OMRAT',
                Qgis.Success
            )
            return True

        except Exception as e:
            self.exception = e
            self.error_msg = str(e)
            QgsMessageLog.logMessage(
                f'Calculation failed with error: {self.error_msg}',
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
            # Success - emit signal with calculation object
            self.calculation_finished.emit(self.calc)
            QgsMessageLog.logMessage(
                'Calculation task finished successfully',
                'OMRAT',
                Qgis.Success
            )
        elif self.isCanceled():
            # Cancelled by user
            QgsMessageLog.logMessage(
                'Calculation task was cancelled',
                'OMRAT',
                Qgis.Warning
            )
        else:
            # Failed with error
            error_msg = self.error_msg or "Unknown error"
            self.calculation_failed.emit(error_msg)
            QgsMessageLog.logMessage(
                f'Calculation task failed: {error_msg}',
                'OMRAT',
                Qgis.Critical
            )

    def cancel(self) -> None:
        """
        Called when the user cancels the task.

        This sets the cancellation flag which is checked during the calculation.
        """
        QgsMessageLog.logMessage(
            'Cancelling calculation task...',
            'OMRAT',
            Qgis.Warning
        )
        super().cancel()
