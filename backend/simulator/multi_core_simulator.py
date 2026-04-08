from typing import List, Dict, Optional

from .process import Process
from .core import Core
from .scheduler_base import SchedulerBase


class MultiCoreSimulator:
    """
    Discrete-time multi-core CPU simulator.

    Supports two execution modes:

    1) Full simulation
       simulator.run()

    2) Step-based simulation (for RL)
       simulator.reset()
       simulator.step()
    """

    def __init__(
        self,
        processes: List[Process],
        scheduler: SchedulerBase,
        num_cores: int
    ):

        self.original_processes = processes
        self.scheduler = scheduler
        self.num_cores = num_cores

        self.cores: List[Core] = [Core(i) for i in range(num_cores)]

        self.processes: List[Process] = []
        self.ready_queue: List[Process] = []
        self.completed_processes: List[Process] = []

        self.time: int = 0

        # For visualization
        self.gantt_chart: Dict[int, List[Optional[int]]] = {
            i: [] for i in range(num_cores)
        }

    # ------------------------------------------------------------

    def reset(self):
        """
        Reset simulator state.
        Used for RL episodes or repeated experiments.
        """

        self.time = 0

        self.processes = [p for p in self.original_processes]

        self.ready_queue = []
        self.completed_processes = []

        for core in self.cores:
            core.reset()

        for process in self.processes:
            process.reset()

        self.gantt_chart = {i: [] for i in range(self.num_cores)}

    # ------------------------------------------------------------

    def _add_new_arrivals(self):
        """
        Move processes arriving at current time to ready queue.
        """

        for process in self.processes:

            if process.arrival_time == self.time:
                self.ready_queue.append(process)

    # ------------------------------------------------------------

    def _all_completed(self):

        return len(self.completed_processes) >= len(self.processes)

    # ------------------------------------------------------------

    def step(self):
        """
        Advance simulation by one timestep.
        Useful for RL environments.
        """

        # 1️⃣ Check arrivals
        self._add_new_arrivals()

        # 2️⃣ Update waiting times
        for process in self.ready_queue:
            process.update_waiting_time(self.time)

        # 3️⃣ Scheduler decision
        decisions = self.scheduler.select_process(
            ready_queue=self.ready_queue,
            cores=self.cores,
            time=self.time
        )

        # 4️⃣ Assign processes to cores
        assigned_pids = set()
        
        # Ensure every core is explicitly accounted for
        final_decisions = {c.core_id: None for c in self.cores}
        for core_id, process in decisions.items():
            if process is not None:
                # Validation: Prevent same process on multiple cores
                if process.pid in assigned_pids:
                    # Duplicate residency detected! Force this assignment to None
                    final_decisions[core_id] = None
                else:
                    final_decisions[core_id] = process
                    assigned_pids.add(process.pid)
            else:
                final_decisions[core_id] = None

        for core_id, process in final_decisions.items():
            # [CRITICAL FIX] If core is preempted, put old process back in ready_queue
            old_process = self.cores[core_id].current_process
            if old_process is not None and old_process != process and not old_process.is_completed():
                if old_process not in self.ready_queue:
                    self.ready_queue.append(old_process)
                    
            self.cores[core_id].assign_process(process)

        # Remove assigned processes from ready queue
        for core in self.cores:
            if core.current_process in self.ready_queue:
                self.ready_queue.remove(core.current_process)

        # 5️⃣ Execute 1 time unit
        completed = []

        for core in self.cores:

            finished = core.execute(self.time, time_slice=1)

            # Gantt logging
            if core.current_process:
                self.gantt_chart[core.core_id].append(
                    core.current_process.pid
                )
            else:
                self.gantt_chart[core.core_id].append(None)

            if finished:
                completed.append(finished)
                self.completed_processes.append(finished)

        # 6️⃣ Advance time
        self.time += 1

        done = self._all_completed()

        return completed, done

    # ------------------------------------------------------------

    def get_state(self):
        """
        Return simulator state (used for RL observation).
        """

        return {
            "time": self.time,
            "ready_queue_length": len(self.ready_queue),
            "running_processes": [
                core.current_process.pid if core.current_process else None
                for core in self.cores
            ],
            "core_utilization": [
                core.busy_time for core in self.cores
            ]
        }

    # ------------------------------------------------------------

    def run(self):
        """
        Run simulation until completion.
        Used for classical schedulers.
        """

        self.reset()

        while not self._all_completed():
            self.step()

        return {
            "processes": self.processes,
            "cores": self.cores,
            "gantt_chart": self.gantt_chart,
            "total_time": self.time
        }