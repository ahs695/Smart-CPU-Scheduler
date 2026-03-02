# backend/simulator/multi_core_simulator.py

from typing import List, Dict, Optional
from .process import Process
from .core import Core
from .scheduler_base import SchedulerBase


class MultiCoreSimulator:
    """
    Pure discrete-time multi-core CPU simulator.

    Responsibilities:
    - Handle arrivals
    - Maintain ready queue
    - Ask scheduler for decisions
    - Execute cores
    - Track completion
    - Record Gantt chart

    Does NOT compute performance metrics.
    Does NOT contain ML or RL logic.
    """

    def __init__(
        self,
        processes: List[Process],
        scheduler: SchedulerBase,
        num_cores: int
    ):
        self.processes = processes
        self.scheduler = scheduler
        self.num_cores = num_cores

        self.cores: List[Core] = [
            Core(core_id=i) for i in range(num_cores)
        ]

        self.ready_queue: List[Process] = []
        self.completed_processes: List[Process] = []

        self.time: int = 0

        # Gantt chart: {core_id: [pid or None per timestep]}
        self.gantt_chart: Dict[int, List[Optional[int]]] = {
            i: [] for i in range(num_cores)
        }

    # ------------------------------------------------------------

    def _add_new_arrivals(self):
        """
        Move processes arriving at current time to ready queue.
        """
        for process in self.processes:
            if process.arrival_time == self.time:
                self.ready_queue.append(process)

    # ------------------------------------------------------------

    def _remove_completed_from_ready(self, process: Process):
        """
        Remove completed process from ready queue if present.
        """
        if process in self.ready_queue:
            self.ready_queue.remove(process)

    # ------------------------------------------------------------

    def _all_completed(self) -> bool:
        return len(self.completed_processes) == len(self.processes)

    # ------------------------------------------------------------

    def run(self) -> Dict:
        """
        Run simulation until all processes complete.

        Returns:
            dict containing raw simulation data.
        """

        # Sort processes by arrival time (deterministic behavior)
        self.processes.sort(key=lambda p: p.arrival_time)

        while not self._all_completed():

            # 1️⃣ Handle arrivals
            self._add_new_arrivals()

            # 2️⃣ Update waiting time for ready processes
            for process in self.ready_queue:
                process.update_waiting_time(self.time)

            # 3️⃣ Ask scheduler for decisions
            # Expected return:
            # { core_id: Process or None }
            scheduling_decisions = self.scheduler.select_process(
                ready_queue=self.ready_queue,
                cores=self.cores,
                time=self.time
            )

            # 4️⃣ Assign processes to cores
            for core_id, process in scheduling_decisions.items():
                self.cores[core_id].assign_process(process)

            # Remove assigned processes from ready queue
            for core in self.cores:
                if core.current_process in self.ready_queue:
                    self.ready_queue.remove(core.current_process)

            # 5️⃣ Execute one time unit per core
            for core in self.cores:

                completed_process = core.execute(
                    current_time=self.time,
                    time_slice=1
                )

                # Log Gantt chart
                if core.current_process:
                    self.gantt_chart[core.core_id].append(
                        core.current_process.pid
                    )
                else:
                    self.gantt_chart[core.core_id].append(None)

                # Handle completion
                if completed_process:
                    self.completed_processes.append(completed_process)
                    self._remove_completed_from_ready(completed_process)

            # 6️⃣ Advance time
            self.time += 1

        return {
            "processes": self.processes,
            "cores": self.cores,
            "gantt_chart": self.gantt_chart,
            "total_time": self.time
        }