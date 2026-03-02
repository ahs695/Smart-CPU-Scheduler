# backend/simulator/traditional/sjf.py

from typing import Dict, List, Optional
from ..scheduler_base import SchedulerBase
from ..process import Process
from ..core import Core


class SJFScheduler(SchedulerBase):
    """
    Shortest Job First Scheduler.

    Supports:
    - Non-preemptive SJF
    - Preemptive SRTF (Shortest Remaining Time First)
    """

    def __init__(self, preemptive: bool = False):
        super().__init__(preemptive=preemptive)

    # ------------------------------------------------------------

    def select_process(
        self,
        ready_queue: List[Process],
        cores: List[Core],
        time: int
    ) -> Dict[int, Optional[Process]]:

        decisions: Dict[int, Optional[Process]] = {}

        if not self.preemptive:
            return self._non_preemptive(ready_queue, cores)
        else:
            return self._preemptive(ready_queue, cores)

    # ------------------------------------------------------------

    def _non_preemptive(
        self,
        ready_queue: List[Process],
        cores: List[Core]
    ) -> Dict[int, Optional[Process]]:

        decisions: Dict[int, Optional[Process]] = {}

        # Sort ready queue by burst time (shortest first)
        ready_queue.sort(key=lambda p: p.burst_time)

        for core in cores:

            # If already running, continue (non-preemptive)
            if core.current_process is not None:
                decisions[core.core_id] = core.current_process
                continue

            # Assign shortest job if available
            if ready_queue:
                next_process = ready_queue.pop(0)
                decisions[core.core_id] = next_process
            else:
                decisions[core.core_id] = None

        return decisions

    # ------------------------------------------------------------

    def _preemptive(
        self,
        ready_queue: List[Process],
        cores: List[Core]
    ) -> Dict[int, Optional[Process]]:

        decisions: Dict[int, Optional[Process]] = {}

        # Collect all runnable processes:
        # ready queue + currently running processes
        runnable_processes = ready_queue.copy()

        for core in cores:
            if core.current_process is not None:
                runnable_processes.append(core.current_process)

        # Sort by remaining time (shortest remaining first)
        runnable_processes.sort(key=lambda p: p.remaining_time)

        # Select up to num_cores shortest processes
        selected = runnable_processes[:len(cores)]

        # Assign shortest processes to cores
        for i, core in enumerate(cores):
            if i < len(selected):
                decisions[core.core_id] = selected[i]
            else:
                decisions[core.core_id] = None

        return decisions

    # ------------------------------------------------------------

    def reset(self):
        """
        No internal state to reset,
        included for framework consistency.
        """
        pass