# backend/simulator/process.py

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Process:
    """
    Process Control Block (PCB) abstraction for CPU Scheduling simulation.
    Designed for classical, CFS-inspired, and AI-driven schedulers.
    """

    pid: int
    arrival_time: int
    burst_time: int
    priority: int = 0

    # Runtime fields (initialized later)
    remaining_time: int = field(init=False)
    completion_time: Optional[int] = None
    start_time: Optional[int] = None
    response_time: Optional[int] = None
    waiting_time: int = 0
    turnaround_time: Optional[int] = None

    # Multi-core + fairness extensions
    current_core: Optional[int] = None
    virtual_runtime: float = 0.0  # For CFS-like fairness
    execution_history: List[int] = field(default_factory=list)

    def __post_init__(self):
        """
        Automatically called after initialization.
        Sets remaining_time equal to total burst_time.
        """
        self.remaining_time = self.burst_time

    # ------------------------------------------------------------
    # STATE TRANSITIONS
    # ------------------------------------------------------------

    def execute(self, current_time: int, time_slice: int = 1):
        """
        Simulates execution of the process for given time_slice.

        Args:
            current_time (int): Current simulation time
            time_slice (int): CPU time allocated

        Returns:
            int: Actual time executed
        """

        if self.start_time is None:
            self.start_time = current_time
            self.response_time = current_time - self.arrival_time

        actual_execution = min(self.remaining_time, time_slice)

        self.remaining_time -= actual_execution
        self.execution_history.append(actual_execution)

        return actual_execution

    # ------------------------------------------------------------

    def update_waiting_time(self, current_time: int):
        """
        Updates waiting time while process is in READY state.
        """
        if self.start_time is None:
            self.waiting_time = current_time - self.arrival_time
        else:
            executed_time = sum(self.execution_history)
            self.waiting_time = current_time - self.arrival_time - executed_time

    # ------------------------------------------------------------

    def complete(self, current_time: int):
        """
        Marks process as completed.
        """
        self.completion_time = current_time
        self.turnaround_time = self.completion_time - self.arrival_time

    # ------------------------------------------------------------

    def is_completed(self) -> bool:
        """
        Returns True if process has finished execution.
        """
        return self.remaining_time == 0

    # ------------------------------------------------------------

    def reset(self):
        """
        Resets runtime metrics for re-running experiments.
        Useful for statistical evaluation across multiple schedulers.
        """
        self.remaining_time = self.burst_time
        self.completion_time = None
        self.start_time = None
        self.response_time = None
        self.waiting_time = 0
        self.turnaround_time = None
        self.current_core = None
        self.virtual_runtime = 0.0
        self.execution_history.clear()

    # ------------------------------------------------------------

    def __repr__(self):
        return (
            f"Process(pid={self.pid}, arrival={self.arrival_time}, "
            f"burst={self.burst_time}, remaining={self.remaining_time})"
        )
        
