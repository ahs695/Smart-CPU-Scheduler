# backend/simulator/traditional/fcfs.py

from typing import Dict, List, Optional
from ..scheduler_base import SchedulerBase
from ..process import Process
from ..core import Core


class FCFSScheduler(SchedulerBase):
    """
    First Come First Serve (FCFS) Scheduler.

    Characteristics:
    - Non-preemptive
    - FIFO order
    - Multi-core aware
    """

    def __init__(self):
        super().__init__(preemptive=False)

    # ------------------------------------------------------------

    def select_process(
        self,
        ready_queue: List[Process],
        cores: List[Core],
        time: int
    ) -> Dict[int, Optional[Process]]:
        """
        Select processes for each core using FCFS policy.
        """

        decisions: Dict[int, Optional[Process]] = {}

        # Sort ready queue by arrival time to enforce FIFO order
        # Stable sort preserves order among same-arrival processes
        ready_queue.sort(key=lambda p: p.arrival_time)

        for core in cores:

            # If core already running a process (non-preemptive),
            # continue running it
            if core.current_process is not None:
                decisions[core.core_id] = core.current_process
                continue

            # If core is idle, assign next process in FIFO order
            if ready_queue:
                next_process = ready_queue.pop(0)
                decisions[core.core_id] = next_process
            else:
                decisions[core.core_id] = None

        return decisions

    # ------------------------------------------------------------

    def reset(self):
        """
        FCFS has no internal state,
        but method included for framework consistency.
        """
        pass