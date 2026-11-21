# backend/simulator/traditional/edf.py

from typing import Dict, List, Optional

from ..scheduler_base import SchedulerBase, assign_keeping_cores, runnable_processes
from ..process import Process
from ..core import Core


class EDFScheduler(SchedulerBase):
    """
    Earliest Deadline First (global, preemptive).

    Every tick the `num_cores` runnable processes with the earliest absolute
    deadlines run. On one core EDF is optimal: if any schedule meets every
    deadline, EDF does. Processes without a deadline run only when no
    deadline-bound work is waiting (in arrival order).
    """

    def __init__(self):
        super().__init__(preemptive=True)

    @staticmethod
    def _key(p: Process):
        no_deadline = p.deadline is None
        return (no_deadline, p.deadline if not no_deadline else 0, p.arrival_time, p.pid)

    def select_process(
        self,
        ready_queue: List[Process],
        cores: List[Core],
        time: int
    ) -> Dict[int, Optional[Process]]:
        runnable = sorted(runnable_processes(ready_queue, cores), key=self._key)
        return assign_keeping_cores(runnable[: len(cores)], cores)
