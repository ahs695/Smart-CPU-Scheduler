# backend/simulator/scheduler_base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from .process import Process
from .core import Core


class SchedulerBase(ABC):
    """
    Abstract Base Class for all scheduling algorithms.

    Any scheduling policy (FCFS, SJF, RR, MLFQ, RL-based, Hybrid)
    must inherit from this class and implement select_process().
    """

    def __init__(self, preemptive: bool = True):
        """
        Args:
            preemptive (bool): Whether scheduler supports preemption.
        """
        self.preemptive = preemptive

    # ------------------------------------------------------------

    @abstractmethod
    def select_process(
        self,
        ready_queue: List[Process],
        cores: List[Core],
        time: int
    ) -> Dict[int, Optional[Process]]:
        """
        Decide which process should run on each core.

        Args:
            ready_queue (List[Process]):
                Processes currently ready to execute.

            cores (List[Core]):
                List of CPU cores (may already be running processes).

            time (int):
                Current simulation time.

        Returns:
            Dict[int, Optional[Process]]:
                Mapping from core_id to selected Process (or None).
        """
        pass

    # ------------------------------------------------------------

    def on_process_completion(
        self,
        process: Process,
        time: int
    ):
        """
        Optional hook triggered when a process completes.

        Useful for:
        - MLFQ queue adjustments
        - RL reward shaping
        - Logging
        """
        pass

    # ------------------------------------------------------------

    def on_time_step(self, time: int):
        """
        Optional hook called at each time step.

        Useful for:
        - Aging mechanisms
        - Dynamic priority adjustments
        - Quantum updates (RR)
        """
        pass

    # ------------------------------------------------------------

    def reset(self):
        """
        Reset internal scheduler state between simulation runs.

        Important for reproducibility in experiments.
        """
        pass