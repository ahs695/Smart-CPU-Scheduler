# backend/simulator/core.py

from typing import Optional
from .process import Process

class Core:
    """
    Represents a single CPU core in a multi-core system.
    Responsible for executing assigned processes and tracking core-level metrics.
    """

    def __init__(self, core_id: int):
        self.core_id = core_id

        # Currently running process
        self.current_process: Optional[Process] = None

        # Metrics
        self.busy_time: int = 0
        self.idle_time: int = 0
        self.context_switches: int = 0

        # Internal tracking
        self.last_process_id: Optional[int] = None

    # ------------------------------------------------------------
    # PROCESS ASSIGNMENT
    # ------------------------------------------------------------

    def assign_process(self, process: Optional[Process]):
        """
        Assign a process to this core.
        Handles context switch counting.
        """

        # Defensive: refuse to assign a process that has already completed.
        # A completed process sitting on a core would produce a 0-length entry
        # in execution_history on the next tick.
        if process is not None and process.is_completed():
            process = None

        if process is not None:
            process.current_core = self.core_id

        # Context switch detection
        if (
            self.current_process is not None
            and process is not None
            and self.current_process.pid != process.pid
        ):
            self.context_switches += 1

        self.current_process = process

    # ------------------------------------------------------------

    def execute(self, current_time: int, time_slice: int = 1):
        """
        Execute the current process for a given time slice.
        Updates utilization and handles process completion.

        Returns:
            completed_process (Optional[Process])
        """

        if self.current_process is None:
            self.idle_time += time_slice
            return None

        # Defensive: a completed process should never sit on a core at execute().
        # If it does, the underlying Process.execute() would append a 0 to
        # execution_history (min(remaining=0, slice) = 0). Treat as idle and
        # clear the core so the scheduler picks someone else next tick.
        if self.current_process.is_completed():
            self.current_process = None
            self.idle_time += time_slice
            return None

        executed_time = self.current_process.execute(current_time, time_slice)

        self.busy_time += executed_time

        # Update virtual runtime for fairness (CFS-style behavior)
        self.current_process.virtual_runtime += executed_time

        # Check completion
        if self.current_process.is_completed():
            self.current_process.complete(current_time + executed_time)
            completed = self.current_process
            self.current_process = None
            return completed

        return None

    # ------------------------------------------------------------

    def is_idle(self) -> bool:
        """
        Returns True if the core is idle.
        """
        return self.current_process is None

    # ------------------------------------------------------------

    def utilization(self, total_time: int) -> float:
        """
        Returns utilization ratio of the core.
        """
        if total_time == 0:
            return 0.0
        return self.busy_time / total_time

    # ------------------------------------------------------------

    def reset(self):
        """
        Reset core state for new experiment runs.
        """
        self.current_process = None
        self.busy_time = 0
        self.idle_time = 0
        self.context_switches = 0
        self.last_process_id = None

    # ------------------------------------------------------------

    def __repr__(self):
        return (
            f"Core(id={self.core_id}, "
            f"busy={self.busy_time}, "
            f"idle={self.idle_time}, "
            f"context_switches={self.context_switches})"
        )
        