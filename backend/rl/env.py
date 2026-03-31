# backend/rl/env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional

from backend.simulator.multi_core_simulator import MultiCoreSimulator
from backend.simulator.process import Process
from backend.rl.reward import RewardEngine


class SchedulingEnv(gym.Env):
    """
    Gymnasium Environment for CPU Scheduling.
    
    Fixed Logic:
    - Non-preemptive action mapping to prevent thrashing.
    - Explicit state normalization.
    - Proper termination and truncation.
    - Safe fallback for invalid actions.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        processes: List[Process],
        num_cores: int = 2,
        max_queue_size: int = 50,  # Increased fixed size to handle up to 50 processes
        max_steps: int = 5000,     # Safely high max steps
        use_lstm: bool = False,
        lstm_model: Optional[object] = None
    ):
        super().__init__()

        self.original_processes = processes
        self.num_cores = num_cores
        self.max_queue_size = max_queue_size
        self.max_steps = max_steps

        self.use_lstm = use_lstm
        self.lstm_model = lstm_model

        self.simulator = None
        self.reward_engine = RewardEngine()

        self.current_action = 0
        self.steps = 0
        self.invalid_action_penalty = 0.0

        # State vector: [queue_len] + [wait_times] + [rem_times] + [core_util] + [time_progress]
        obs_size = (
            1
            + max_queue_size
            + max_queue_size
            + num_cores
            + 1
        )

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(obs_size,),
            dtype=np.float32
        )

        # Action = index of process in ready queue to schedule
        self.action_space = spaces.Discrete(max_queue_size)

    # ------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Simulator persists across steps for the entire episode
        self.simulator = MultiCoreSimulator(
            processes=self.original_processes,
            scheduler=self,
            num_cores=self.num_cores
        )

        self.simulator.reset()
        self.reward_engine.reset()

        self.steps = 0
        self.invalid_action_penalty = 0.0

        return self._get_state(), {}

    # ------------------------------------------------------------

    def step(self, action):
        self.current_action = int(action)
        self.invalid_action_penalty = 0.0
        
        # Check if the chosen action is valid (out of bounds)
        if len(self.simulator.ready_queue) > 0:
            if self.current_action >= len(self.simulator.ready_queue):
                # Fallback to safe policy (FCFS, i.e., index 0)
                self.current_action = 0
                self.invalid_action_penalty = -0.1  # small penalty for invalid choice

        self.steps += 1

        # Advance simulator by ONE timestep
        completed, sim_all_done = self.simulator.step()

        # State and Reward
        state = self._get_state()
        
        step_reward = self.reward_engine.compute(
            completed=completed,
            ready_queue=self.simulator.ready_queue,
            cores=self.simulator.cores
        )

        reward = step_reward + self.invalid_action_penalty

        # Termination rules
        terminated = sim_all_done
        truncated = self.steps >= self.max_steps
        
        # Severe penalty if we run out of time (didn't make progress)
        if truncated and not terminated:
            incomplete = [
                p for p in self.simulator.processes
                if p.turnaround_time is None
            ]
            reward -= 1.0 * len(incomplete)  # Heavy punishment for hanging

        return state, reward, terminated, truncated, {}

    # ------------------------------------------------------------

    def select_process(self, ready_queue, cores, time):
        """
        Called BY the simulator during step() to get core assignments.
        This handles mapping the action to an available core.
        """
        decisions = {}
        
        if not ready_queue:
            return decisions

        rq = list(ready_queue)
        
        # Find which cores are actually available (prevent thrashing by skipping busy cores)
        idle_cores = [c for c in cores if c.current_process is None]
        
        if not idle_cores:
            # CPU is fully utilized. Do nothing (no preemption).
            return decisions

        # 1️⃣ Primary assignment from agent's action
        # We already ensured current_action is valid in step()
        selected_idx = self.current_action
        
        # Safety bound (just in case)
        if selected_idx >= len(rq):
            selected_idx = 0
            
        selected_proc = rq.pop(selected_idx)
        
        # Assign to the first available core
        target_core = idle_cores.pop(0)
        decisions[target_core.core_id] = selected_proc
        
        # 2️⃣ Auto-fill remaining idle cores with FCFS
        # The prompt requires: "IF ready_queue is NOT empty: CPU must NEVER be idle"
        for core in idle_cores:
            if rq:
                proc = rq.pop(0)  # FCFS
                decisions[core.core_id] = proc
                
        return decisions

    # ------------------------------------------------------------

    def _get_state(self):
        """
        Constructs a normalized, fixed-size state vector.
        """
        rq = self.simulator.ready_queue

        # 1. Queue length (normalized)
        q_len = min(len(rq), self.max_queue_size)
        norm_q_len = q_len / self.max_queue_size

        # 2. Waiting times & Remaining times (padded & normalized)
        waiting = []
        remaining = []
        
        # Assuming maximum expected metrics for normalization:
        # e.g., max wait time 1000, max burst 500
        MAX_WAIT = 1000.0
        MAX_BURST = 500.0

        for i in range(self.max_queue_size):
            if i < len(rq):
                w = min(rq[i].waiting_time / MAX_WAIT, 1.0)
                r = min(rq[i].remaining_time / MAX_BURST, 1.0)
                waiting.append(w)
                remaining.append(r)
            else:
                waiting.append(0.0)
                remaining.append(0.0)

        # 3. Core Utilization (1 if busy, 0 if idle)
        core_busy = [
            1.0 if core.current_process else 0.0
            for core in self.simulator.cores
        ]

        # 4. Time progress
        time_norm = min(self.simulator.time / self.max_steps, 1.0)

        state = np.array(
            [norm_q_len] + waiting + remaining + core_busy + [time_norm],
            dtype=np.float32
        )

        return state

    # ------------------------------------------------------------

    def render(self):
        pass