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

# ------------------------------------------------------------
# Shared helpers for priority-based (top-N) schedulers
# ------------------------------------------------------------

def runnable_processes(ready_queue: List[Process], cores: List[Core]) -> List[Process]:
    """Running processes plus ready ones, without duplicates or finished processes."""
    running = [c.current_process for c in cores if c.current_process is not None]
    running_ids = {id(p) for p in running}
    return running + [p for p in ready_queue if id(p) not in running_ids and not p.is_completed()]


def assign_keeping_cores(selected: List[Process], cores: List[Core]) -> Dict[int, Optional[Process]]:
    """
    Map the processes chosen to run onto cores. A selected process that is
    already running keeps its core (no needless migration or context switch);
    newcomers fill idle cores first, then cores whose process was preempted.
    """
    decisions = {core.core_id: core.current_process for core in cores}
    selected_ids = {id(p) for p in selected}
    running_ids = {id(c.current_process) for c in cores if c.current_process is not None}

    free_cores = [
        core for core in cores
        if core.current_process is None or id(core.current_process) not in selected_ids
    ]
    newcomers = [p for p in selected if id(p) not in running_ids]
    free_cores.sort(key=lambda c: c.current_process is not None)
    for core in free_cores:
        decisions[core.core_id] = newcomers.pop(0) if newcomers else None
    return decisions
