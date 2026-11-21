# backend/simulator/traditional/rms.py

from typing import Dict, List, Optional

from ..scheduler_base import SchedulerBase, assign_keeping_cores, runnable_processes
from ..process import Process
from ..core import Core


class RMSScheduler(SchedulerBase):
    """
    Rate Monotonic Scheduling (global, preemptive, fixed priority).

    Jobs of periodic tasks get a static priority from their task's period:
    the shorter the period, the higher the priority. A single-core task set is
    guaranteed schedulable when its utilisation is at most n(2^(1/n) - 1)
    (Liu & Layland). Processes without a period come last, in arrival order.
    """

    def __init__(self):
        super().__init__(preemptive=True)

    @staticmethod
    def _key(p: Process):
        no_period = p.period is None
        return (no_period, p.period if not no_period else 0, p.arrival_time, p.pid)

    def select_process(
        self,
        ready_queue: List[Process],
        cores: List[Core],
        time: int
    ) -> Dict[int, Optional[Process]]:
        runnable = sorted(runnable_processes(ready_queue, cores), key=self._key)
        return assign_keeping_cores(runnable[: len(cores)], cores)
