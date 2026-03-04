# backend/simulator/traditional/round_robin.py

from typing import Dict, List, Optional
from ..scheduler_base import SchedulerBase
from ..process import Process
from ..core import Core


class RoundRobinScheduler(SchedulerBase):
    """
    Round Robin Scheduler.

    Characteristics:
    - Preemptive
    - Time quantum based
    - Fair scheduling
    """

    def __init__(self, quantum: int = 2):
        super().__init__(preemptive=True)
        self.quantum = quantum

        # Tracks how much quantum a process has used
        self.time_slice_used: Dict[int, int] = {}

    # ------------------------------------------------------------

    def select_process(
        self,
        ready_queue: List[Process],
        cores: List[Core],
        time: int
    ) -> Dict[int, Optional[Process]]:

        decisions: Dict[int, Optional[Process]] = {}

        for core in cores:

            current = core.current_process

            # If core is running process
            if current is not None:

                pid = current.pid

                if pid not in self.time_slice_used:
                    self.time_slice_used[pid] = 0

                self.time_slice_used[pid] += 1

                # If quantum exhausted → preempt
                if self.time_slice_used[pid] >= self.quantum:
                    self.time_slice_used[pid] = 0

                    # Move process back to ready queue if not finished
                    if not current.is_completed():
                        ready_queue.append(current)

                    # Assign next process
                    if ready_queue:
                        next_process = ready_queue.pop(0)
                        decisions[core.core_id] = next_process
                    else:
                        decisions[core.core_id] = None

                else:
                    # Continue running current process
                    decisions[core.core_id] = current

            else:
                # Core idle → assign next ready process
                if ready_queue:
                    next_process = ready_queue.pop(0)
                    decisions[core.core_id] = next_process
                else:
                    decisions[core.core_id] = None

        return decisions

    # ------------------------------------------------------------

    def reset(self):
        """
        Reset scheduler state between simulations.
        """
        self.time_slice_used.clear()