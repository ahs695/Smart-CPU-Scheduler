# backend/rl/env.py
"""
Gymnasium environment for learned multi-core CPU scheduling.

The agent learns a *priority function*. Every tick it outputs one weight per
process feature; each runnable process (ready or running) is scored as
w · features, and the `num_cores` highest-scoring processes run. Processes
that stay selected keep their core, so nothing is switched needlessly.

Why this formulation:
  - The action means the same thing for any queue length or core count, and it
    can express FCFS (weight on age), least-attained-service (negative weight
    on CPU time used), and with predictions an SRTF-like policy.
  - The policy sees the system state, so the weights can change with load.
  - It is work-conserving: a core is never left idle while a process is runnable
    (idling never improves turnaround or waiting time).

Information model — realistic, like a real OS:
  The agent never sees a process's true burst or remaining time. Per-process
  features are only what a scheduler could observe: waiting time, CPU time
  received so far, age, and whether it is currently running. The Hybrid
  subclass adds the LSTM's *predicted* burst. SJF/SRTF, which do use true
  bursts, act as oracle upper bounds in the experiments.
"""

from typing import Callable, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from backend.simulator.multi_core_simulator import MultiCoreSimulator
from backend.simulator.process import Process
from backend.simulator.workload_generator import sample_training_workload
from backend.rl.reward import RewardEngine

TIME_SCALE = 50.0   # ticks -> feature units
LOG_SCALE = 5.0     # log1p(ticks) / LOG_SCALE keeps log features ~[0, 1.5]
STAT_FNS = (np.min, np.mean, np.max, np.std)
N_GLOBAL = 9
SHORT_JOB = 10      # ticks; "short" threshold for the completed-job statistics


class SchedulingEnv(gym.Env):
    """
    Args:
        processes: Fixed workload to schedule (app / evaluation). When None,
            every reset() samples a fresh randomised training workload.
        num_cores: Number of simulated cores.
        max_queue_size: Unused; kept for backwards-compatible construction.
        max_steps: Episode truncation limit (safety net).
        reward_engine: Optional custom RewardEngine.
    """

    metadata = {"render_modes": []}

    BASE_FEATURES = [
        "waiting", "attained", "age", "running",
        "log_waiting", "log_attained", "log_age",
    ]

    def __init__(
        self,
        processes: Optional[List[Process]] = None,
        num_cores: int = 2,
        max_queue_size: int = 50,
        max_steps: int = 15000,
        reward_engine: Optional[RewardEngine] = None,
        workload_sampler: Optional[Callable] = None,
    ):
        super().__init__()

        self.original_processes = processes
        self.num_cores = num_cores
        self.max_steps = max_steps
        self.reward_engine = reward_engine or RewardEngine()
        self.workload_sampler = workload_sampler or sample_training_workload

        self.feature_names = self.BASE_FEATURES + self._extra_feature_names()
        self.n_features = len(self.feature_names)

        # Global: runnable/cores, ready-queue length, busy fraction, cores/8, recent arrivals,
        # + statistics of jobs that already finished (count, mean, std, short fraction)
        # + min/mean/max/std of every feature over ready processes and over running processes
        obs_size = N_GLOBAL + 2 * len(STAT_FNS) * self.n_features
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_features,), dtype=np.float32)

        self.simulator: Optional[MultiCoreSimulator] = None
        self.current_action = np.zeros(self.n_features, dtype=np.float32)
        self.steps = 0

    # ------------------------------------------------------------
    # Hooks for the Hybrid subclass
    # ------------------------------------------------------------

    def _extra_feature_names(self) -> List[str]:
        return []

    def _extra_features(self, procs: List[Process], attained: np.ndarray) -> Optional[np.ndarray]:
        return None

    def _on_reset(self):
        pass

    # ------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.original_processes is not None:
            processes = self.original_processes
        else:
            processes = self.workload_sampler(self.np_random, self.num_cores)

        self.processes = processes
        self.arrivals = sorted(p.arrival_time for p in processes)
        self.simulator = MultiCoreSimulator(processes=processes, scheduler=self, num_cores=self.num_cores)
        self.simulator.reset()
        self.steps = 0
        self._on_reset()

        return self._get_state(), {}

    def step(self, action):
        self.current_action = np.asarray(action, dtype=np.float32).reshape(-1)
        sim = self.simulator

        tick = sim.time
        completed_before = len(sim.completed_processes)
        switches_before = sum(c.context_switches for c in sim.cores)

        _, sim_done = sim.step()
        self.steps += 1

        # Jobs present during this tick (arrived by `tick`, not completed before it)
        jobs_in_system = int(np.searchsorted(self.arrivals, tick, side="right")) - completed_before
        switches = sum(c.context_switches for c in sim.cores) - switches_before
        reward = self.reward_engine.compute(jobs_in_system, switches)

        terminated = sim_done
        truncated = (not terminated) and self.steps >= self.max_steps
        return self._get_state(), reward, terminated, truncated, {}

    # ------------------------------------------------------------
    # Scheduling decision (called by MultiCoreSimulator.step)
    # ------------------------------------------------------------

    def select_process(self, ready_queue, cores, time):
        decisions = {core.core_id: core.current_process for core in cores}

        running = [c.current_process for c in cores if c.current_process is not None]
        running_ids = {id(p) for p in running}
        runnable = running + [p for p in ready_queue if id(p) not in running_ids and not p.is_completed()]
        if not runnable:
            return decisions

        feats = self._features(runnable, time, running_ids)
        scores = feats @ self.current_action
        # Deterministic tie-break: earlier arrival, then lower pid
        order = sorted(
            range(len(runnable)),
            key=lambda i: (-scores[i], runnable[i].arrival_time, runnable[i].pid),
        )
        selected = [runnable[i] for i in order[: len(cores)]]
        selected_ids = {id(p) for p in selected}

        # Keep selected processes on the core they already occupy
        free_cores = []
        for core in cores:
            if core.current_process is not None and id(core.current_process) in selected_ids:
                continue
            free_cores.append(core)
        newcomers = [p for p in selected if id(p) not in running_ids]
        # Fill idle cores first so preemption only happens when needed
        free_cores.sort(key=lambda c: c.current_process is not None)
        for core in free_cores:
            decisions[core.core_id] = newcomers.pop(0) if newcomers else None

        return decisions

    # ------------------------------------------------------------
    # Features and observation
    # ------------------------------------------------------------

    def _features(self, procs: List[Process], time: int, running_ids) -> np.ndarray:
        attained = np.array([p.burst_time - p.remaining_time for p in procs], dtype=np.float32)
        waiting = np.array([p.waiting_time for p in procs], dtype=np.float32)
        age = np.array([time - p.arrival_time for p in procs], dtype=np.float32)
        is_running = np.array([1.0 if id(p) in running_ids else 0.0 for p in procs], dtype=np.float32)

        cols = [
            waiting / TIME_SCALE,
            attained / TIME_SCALE,
            age / TIME_SCALE,
            is_running,
            np.log1p(waiting) / LOG_SCALE,
            np.log1p(attained) / LOG_SCALE,
            np.log1p(age) / LOG_SCALE,
        ]
        extra = self._extra_features(procs, attained)
        if extra is not None:
            cols.extend(extra)
        return np.stack(cols, axis=1)

    def _stats(self, feats: np.ndarray) -> List[float]:
        if feats.shape[0] == 0:
            return [0.0] * (len(STAT_FNS) * self.n_features)
        return np.concatenate([fn(feats, axis=0) for fn in STAT_FNS]).tolist()

    def _get_state(self) -> np.ndarray:
        sim = self.simulator
        time = sim.time
        running = [c.current_process for c in sim.cores if c.current_process is not None]
        running_ids = {id(p) for p in running}
        ready = [p for p in sim.ready_queue if id(p) not in running_ids and not p.is_completed()]
        # Processes arriving exactly now join the ready queue at the start of the next step
        ready += [p for p in self.processes if p.arrival_time == time]

        ready_feats = self._features(ready, time, running_ids) if ready else np.zeros((0, self.n_features))
        run_feats = self._features(running, time, running_ids) if running else np.zeros((0, self.n_features))

        recent = int(np.searchsorted(self.arrivals, time, side="right")
                     - np.searchsorted(self.arrivals, time - 20, side="right"))
        # Lengths of finished jobs are observable history (a real OS knows them);
        # they let the policy infer the workload's burst distribution.
        done = [p.burst_time for p in sim.completed_processes]
        if done:
            done_stats = [min(len(done) / 30.0, 2.0), np.mean(done) / TIME_SCALE,
                          np.std(done) / TIME_SCALE, sum(b < SHORT_JOB for b in done) / len(done)]
        else:
            done_stats = [0.0, 0.0, 0.0, 0.0]

        globals_ = [
            min((len(ready) + len(running)) / (4.0 * self.num_cores), 3.0),
            min(len(ready) / 20.0, 3.0),
            len(running) / self.num_cores,
            self.num_cores / 8.0,
            recent / 10.0,
        ] + done_stats
        return np.array(globals_ + self._stats(ready_feats) + self._stats(run_feats), dtype=np.float32)

    def render(self):
        pass
