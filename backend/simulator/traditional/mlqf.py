# backend/simulator/traditional/mlfq.py

from typing import Dict, List, Optional
from collections import deque

from ..scheduler_base import SchedulerBase
from ..process import Process
from ..core import Core


class MLFQScheduler(SchedulerBase):
    """
    Multi-Level Feedback Queue Scheduler.

    Features:
    - Multiple priority queues
    - Different quantum per queue
    - Demotion on quantum exhaustion
    - Aging for starvation prevention
    """

    def __init__(
        self,
        num_levels: int = 3,
        quantums: List[int] = None,
        aging_threshold: int = 10
    ):
        super().__init__(preemptive=True)

        self.num_levels = num_levels
        self.quantums = quantums if quantums else [2, 4, 8]
        self.aging_threshold = aging_threshold

        # Priority queues
        self.queues = [deque() for _ in range(self.num_levels)]

        # Process metadata
        self.process_level: Dict[int, int] = {}
        self.time_used: Dict[int, int] = {}
        self.wait_time: Dict[int, int] = {}

    # ------------------------------------------------------------

    def _add_new_processes(self, ready_queue: List[Process]):

        for process in ready_queue:

            pid = process.pid

            if pid not in self.process_level:

                self.process_level[pid] = 0
                self.time_used[pid] = 0
                self.wait_time[pid] = 0

                self.queues[0].append(process)

        ready_queue.clear()

    # ------------------------------------------------------------

    def _apply_aging(self):

        for level in range(1, self.num_levels):

            for process in list(self.queues[level]):

                pid = process.pid
                self.wait_time[pid] += 1

                if self.wait_time[pid] >= self.aging_threshold:

                    self.queues[level].remove(process)

                    new_level = max(0, level - 1)
                    self.process_level[pid] = new_level

                    self.queues[new_level].append(process)

                    self.wait_time[pid] = 0

    # ------------------------------------------------------------

    def _get_next_process(self) -> Optional[Process]:

        for level in range(self.num_levels):
            if self.queues[level]:
                return self.queues[level].popleft()

        return None

    # ------------------------------------------------------------

    def select_process(
        self,
        ready_queue: List[Process],
        cores: List[Core],
        time: int
    ) -> Dict[int, Optional[Process]]:

        decisions: Dict[int, Optional[Process]] = {}

        # Add new arrivals to highest priority queue
        self._add_new_processes(ready_queue)

        # Apply starvation prevention
        self._apply_aging()

        for core in cores:

            current = core.current_process

            if current is not None:

                pid = current.pid
                level = self.process_level[pid]

                self.time_used[pid] += 1

                # Check quantum expiration
                if self.time_used[pid] >= self.quantums[level]:

                    self.time_used[pid] = 0

                    # Demote process if not already lowest level
                    new_level = min(level + 1, self.num_levels - 1)
                    self.process_level[pid] = new_level

                    if not current.is_completed():
                        self.queues[new_level].append(current)

                    next_process = self._get_next_process()

                    decisions[core.core_id] = next_process

                else:
                    decisions[core.core_id] = current

            else:

                next_process = self._get_next_process()
                decisions[core.core_id] = next_process

        return decisions

    # ------------------------------------------------------------

    def reset(self):

        self.queues = [deque() for _ in range(self.num_levels)]
        self.process_level.clear()
        self.time_used.clear()
        self.wait_time.clear()