# backend/rl/env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional

from backend.simulator.multi_core_simulator import MultiCoreSimulator
from backend.simulator.process import Process
from backend.rl.reward import RewardEngine


class SchedulingEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        processes: List[Process],
        num_cores: int = 2,
        max_queue_size: int = 10,
        max_steps: int = 500,   # IMPORTANT
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

        obs_size = (
            1
            + max_queue_size
            + max_queue_size
            + num_cores
        )

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(obs_size,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(max_queue_size)

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

        return self._get_state(), {}

    # ------------------------------------------------------------

    def step(self, action):

        self.current_action = int(action)
        self.steps += 1

        completed, done = self.simulator.step()

        reward = self.reward_engine.compute(
            completed=completed,
            ready_queue=self.simulator.ready_queue,
            cores=self.simulator.cores
        )

        state = self._get_state()

        terminated = done
        truncated = self.steps >= self.max_steps
        
        if terminated or truncated:
            incomplete = [
                p for p in self.simulator.processes
                if p.turnaround_time is None
            ]
            reward -= 20 * len(incomplete)

        return state, reward, terminated, truncated, {}

    # ------------------------------------------------------------

    def select_process(self, ready_queue, cores, time):

        decisions = {}

        for core in cores:

            if not ready_queue:
                decisions[core.core_id] = None
                continue

            idx = min(self.current_action, len(ready_queue) - 1)
            selected = ready_queue[idx]

            # ✅ PREEMPTION (CRITICAL FIX)
            if core.current_process != selected:
                core.context_switches += 1

            decisions[core.core_id] = selected

        return decisions

    # ------------------------------------------------------------

    def _get_state(self):

        rq = self.simulator.ready_queue

        # Normalize EVERYTHING (CRITICAL)
        queue_len = len(rq) / self.max_queue_size

        waiting = [p.waiting_time / 100 for p in rq[:self.max_queue_size]]
        waiting += [0] * (self.max_queue_size - len(waiting))

        remaining = [p.remaining_time / 100 for p in rq[:self.max_queue_size]]
        remaining += [0] * (self.max_queue_size - len(remaining))

        core_busy = [
            1.0 if core.current_process else 0.0
            for core in self.simulator.cores
        ]

        state = np.array(
            [queue_len] + waiting + remaining + core_busy,
            dtype=np.float32
        )

        return state

    # ------------------------------------------------------------

    def render(self):
        pass