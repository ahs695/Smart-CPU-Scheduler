# backend/rl/env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List

from backend.simulator.multi_core_simulator import MultiCoreSimulator
from backend.simulator.process import Process
from backend.rl.reward import RewardEngine


class SchedulingEnv(gym.Env):
    """
    Gymnasium Environment for CPU Scheduling.
    
    Dynamic Preemptive Multi-Core Logic:
    - MultiDiscrete action mapping (per core).
    - True preemption natively supported.
    - Extensive state normalization and padding.
    - Delta-based tracking.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        processes: List[Process],
        num_cores: int = 2,
        max_queue_size: int = 50,
        max_steps: int = 15000
    ):
        super().__init__()

        self.original_processes = processes
        self.num_cores = num_cores
        self.max_queue_size = max_queue_size
        self.max_steps = max_steps

        # Maximum possible selectable processes: max_queue_size + num_cores
        self.max_choices = max_queue_size + num_cores

        self.simulator = None
        self.reward_engine = RewardEngine()

        self.steps = 0
        self.invalid_action_penalty = 0.0

        # State vector per process (padded up to self.max_choices):
        # [remaining_time, waiting_time, arrival_time]
        # + global state:
        # [queue_len, core0_status, core1_status ..., time_progress]
        obs_size = (
            self.max_choices * 3
            + 1  # queue len
            + self.num_cores
            + 1  # time progress
        )

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(obs_size,),
            dtype=np.float32
        )

        # Action: MultiDiscrete [max_choices] for each core
        self.action_space = spaces.MultiDiscrete([self.max_choices] * self.num_cores)
        
        # Delta-reward trackers
        self.prev_total_waiting = 0.0

    # ------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.simulator = MultiCoreSimulator(
            processes=self.original_processes,
            scheduler=self,
            num_cores=self.num_cores
        )

        self.simulator.reset()
        self.reward_engine.reset()

        self.steps = 0
        self.invalid_action_penalty = 0.0
        self.prev_total_waiting = 0.0

        return self._get_state(), {}

    # ------------------------------------------------------------

    def step(self, action):
        self.current_action = action  # It's an array for multi-discrete
        self.invalid_action_penalty = 0.0
        
        # Get wait time BEFORE simulation step for delta calculation
        ready_processes = self.simulator.ready_queue + [c.current_process for c in self.simulator.cores if c.current_process]
        self.prev_total_waiting = sum(p.waiting_time for p in ready_processes) if ready_processes else 0.0

        self.steps += 1

        # Advance simulator by ONE timestep
        completed, sim_all_done = self.simulator.step()

        # Get wait time AFTER
        new_ready = self.simulator.ready_queue + [c.current_process for c in self.simulator.cores if c.current_process]
        curr_total_waiting = sum(p.waiting_time for p in new_ready) if new_ready else 0.0
        
        # State and Reward
        state = self._get_state()
        
        step_reward = self.reward_engine.compute(
            completed=completed,
            ready_queue=new_ready,
            cores=self.simulator.cores,
            prev_wait=self.prev_total_waiting,
            curr_wait=curr_total_waiting,
            selected_preds=[]  # Handled natively as empty for base PPO
        )

        reward = step_reward + self.invalid_action_penalty

        terminated = sim_all_done
        truncated = self.steps >= self.max_steps
        
        if truncated and not terminated:
            incomplete = [
                p for p in self.simulator.processes
                if p.turnaround_time is None
            ]
            reward -= 2.0 * len(incomplete)

        return state, reward, terminated, truncated, {}

    # ------------------------------------------------------------

    def select_process(self, ready_queue, cores, time):
        """
        Extracts action array and maps precisely to each core.
        Enforces residency validation: One process, one core.
        """
        # Ensure every core has an entry (default to None/Idle)
        decisions = {core.core_id: None for core in cores}
        
        # Available pool = queue + currently running
        pool = list(ready_queue)
        for c in cores:
            if c.current_process and c.current_process not in pool:
                pool.append(c.current_process)
                
        if not pool:
            return decisions
            
        assigned_pids = set()

        for i, core in enumerate(cores):
            idx = int(self.current_action[i])
            
            # Validation 1: Out of bounds
            if idx >= len(pool):
                self.invalid_action_penalty -= 0.1
                # Fallback: Keep current, else FCFS
                if core.current_process and core.current_process.pid not in assigned_pids:
                    decisions[core.core_id] = core.current_process
                    assigned_pids.add(core.current_process.pid)
                else:
                    # Find first FCFS that isn't assigned
                    for p in pool:
                        if p.pid not in assigned_pids:
                            decisions[core.core_id] = p
                            assigned_pids.add(p.pid)
                            break
                continue
                
            selected_process = pool[idx]
            
            # Validation 2: Duplicate residency attempt
            if selected_process.pid in assigned_pids:
                self.invalid_action_penalty -= 0.2 # Heavier penalty for duplication
                # Fallback to keep core alive if possible with unique work
                if core.current_process and core.current_process.pid not in assigned_pids:
                    decisions[core.core_id] = core.current_process
                    assigned_pids.add(core.current_process.pid)
                else:
                    for p in pool:
                        if p.pid not in assigned_pids:
                            decisions[core.core_id] = p
                            assigned_pids.add(p.pid)
                            break
                continue

            # Valid assignment!
            decisions[core.core_id] = selected_process
            assigned_pids.add(selected_process.pid)
            
        return decisions

    # ------------------------------------------------------------

    def _get_state(self):
        pool = list(self.simulator.ready_queue)
        for c in self.simulator.cores:
            if c.current_process and c.current_process not in pool:
                pool.append(c.current_process)

        # 1. Per-process states
        process_features = []
        MAX_WAIT = 1000.0
        MAX_BURST = 50.0
        MAX_TIME = float(self.max_steps)

        for i in range(self.max_choices):
            if i < len(pool):
                p = pool[i]
                rem = min(p.remaining_time / MAX_BURST, 1.0)
                wait = min(p.waiting_time / MAX_WAIT, 1.0)
                arr = min(p.arrival_time / MAX_TIME, 1.0)
                process_features.extend([rem, wait, arr])
            else:
                process_features.extend([0.0, 0.0, 0.0])

        # 2. Global states
        q_len = min(len(self.simulator.ready_queue), self.max_queue_size) / self.max_queue_size
        
        core_busy = [
            1.0 if core.current_process else 0.0
            for core in self.simulator.cores
        ]
        
        time_norm = min(self.simulator.time / self.max_steps, 1.0)

        # Concatenate
        state_list = process_features + [q_len] + core_busy + [time_norm]

        state = np.array(state_list, dtype=np.float32)

        return state

    # ------------------------------------------------------------

    def render(self):
        pass