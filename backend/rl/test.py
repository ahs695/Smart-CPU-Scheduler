# backend/rl/test.py — smoke test for the scheduling environments.
#   python -m backend.rl.test
import copy

import numpy as np
from gymnasium.utils.env_checker import check_env

from backend.experiments.workload_factory import generate_workload
from backend.rl.env import SchedulingEnv
from backend.hybrid.hybrid_scheduler import HybridSchedulingEnv
from backend.ml.lstm_model import load_lstm
from backend.paths import LSTM_MODEL_PATH

lstm = load_lstm(LSTM_MODEL_PATH)
procs = generate_workload("mixed", 20, seed=0)

for name, make in [
    ("PPO", lambda p=None: SchedulingEnv(processes=p, num_cores=2)),
    ("Hybrid", lambda p=None: HybridSchedulingEnv(processes=p, num_cores=2, lstm_model=lstm)),
]:
    check_env(make(), skip_render_check=True)            # Gymnasium API compliance
    env = make(copy.deepcopy(procs))
    state, _ = env.reset(seed=0)
    total, done = 0.0, False
    while not done:
        state, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(state)
        total += reward
        done = terminated or truncated
    finished = sum(p.turnaround_time is not None for p in env.simulator.processes)
    assert terminated and finished == len(procs), f"{name}: only {finished}/{len(procs)} finished"
    print(f"{name}: OK — {finished} processes finished in {env.simulator.time} ticks, return {total:.1f}")

print("Env test passed")
