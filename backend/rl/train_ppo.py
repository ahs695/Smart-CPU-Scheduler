# backend/rl/train_ppo.py
"""
Train the PPO (observable features only) and Hybrid (PPO + LSTM predictions)
schedulers.

    python -m backend.rl.train_ppo --mode all --cores 1 2 4 8 --timesteps 2000000

Every episode uses a fresh randomised workload (sample_training_workload).
Rewards are normalised with VecNormalize during training only — observations
are pre-scaled in the env, so the saved policy needs no extra statistics at
inference. A validation callback scores the deterministic policy on fixed
held-out workloads and keeps the best checkpoint.
"""

import argparse
import copy
import csv
import os
import time
from typing import Dict, List

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecMonitor

from backend.rl.env import SchedulingEnv
from backend.hybrid.hybrid_scheduler import HybridSchedulingEnv
from backend.ml.lstm_model import load_lstm
from backend.paths import LSTM_MODEL_PATH, MODELS_DIR, model_path
from backend.experiments.workload_factory import generate_workload, WORKLOAD_TYPES
from backend.simulator.workload_generator import sample_training_workload
from backend.simulator.multi_core_simulator import MultiCoreSimulator
from backend.simulator.metrics import MetricsEngine
from backend.simulator.traditional.fcfs import FCFSScheduler
from backend.simulator.traditional.edf import EDFScheduler
from backend.simulator.realtime import assign_deadlines, expand_periodic, hyperperiod, random_task_set

LOG_DIR = os.path.join(MODELS_DIR, "training_logs")
VAL_SEEDS = range(500, 506)           # fixed-type validation workloads (test seeds are 0-9)
VAL_RANDOM_SEED = 7_777               # seed for validation draws from the training distribution
VAL_RANDOM_COUNT = 40


def output_path(mode: str, num_cores: int, out_dir: str = None) -> str:
    """Path without .zip, as SB3 expects: models/ppo_scheduler[_hybrid]_{N}c"""
    base = "ppo_scheduler" if mode == "ppo" else f"ppo_scheduler_{mode}"
    name = f"{base}_{num_cores}c"
    return os.path.join(out_dir, name) if out_dir else model_path(name)


def make_env(mode: str, num_cores: int, processes=None, lstm=None):
    if mode == "hybrid":
        return HybridSchedulingEnv(processes=processes, num_cores=num_cores, lstm_model=lstm)
    return SchedulingEnv(processes=processes, num_cores=num_cores)


def _env_factory(mode: str, num_cores: int, rank: int, seed: int):
    def _init():
        torch.set_num_threads(1)
        lstm = load_lstm(LSTM_MODEL_PATH) if mode == "hybrid" else None
        env = make_env(mode, num_cores, lstm=lstm)
        env.reset(seed=seed + rank)
        return env
    return _init


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

def validation_workloads(num_cores: int) -> List:
    """Turnaround validation set: fixed workload types (no deadlines) + plain draws from the training mix."""
    workloads = [generate_workload(t, 30, s) for t in WORKLOAD_TYPES for s in VAL_SEEDS]
    rng = np.random.default_rng(VAL_RANDOM_SEED + num_cores)
    while len(workloads) < len(WORKLOAD_TYPES) * len(VAL_SEEDS) + VAL_RANDOM_COUNT:
        w = sample_training_workload(rng, num_cores)
        if all(p.deadline is None for p in w):
            workloads.append(w)
    return workloads


def deadline_validation_workloads(num_cores: int) -> List:
    """Deadline validation set: fixed types with slack-3 deadlines + periodic sets near full load."""
    workloads = []
    for t in WORKLOAD_TYPES:
        for s in VAL_SEEDS[:3]:
            w = generate_workload(t, 30, s)
            assign_deadlines(w, 3.0)
            workloads.append(w)
    rng = np.random.default_rng(VAL_RANDOM_SEED + 100 + num_cores)
    for u in (0.7, 0.8, 0.9, 0.95, 1.0) * 3:
        tasks = random_task_set(int(rng.integers(3, 6)) + 2 * (num_cores - 1), u * num_cores, rng,
                                tolerance=0.05 * num_cores)
        workloads.append(expand_periodic(tasks, min(2 * hyperperiod(tasks), 300)))
    return workloads


def _traditional(scheduler, procs, num_cores) -> Dict:
    sim = MultiCoreSimulator(copy.deepcopy(procs), scheduler, num_cores)
    sim.run()
    return MetricsEngine.summarize(sim.processes, sim.cores, sim.time)


def run_policy(model, mode: str, procs, num_cores: int, lstm=None) -> Dict:
    env = make_env(mode, num_cores, processes=copy.deepcopy(procs), lstm=lstm)
    state, _ = env.reset()
    while True:
        action, _ = model.predict(state, deterministic=True)
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    sim = env.simulator
    return MetricsEngine.summarize(sim.processes, sim.cores, sim.time)


class ValidationCallback(BaseCallback):
    """
    Every `eval_freq` timesteps, run the deterministic policy on two held-out sets:
      - turnaround set: score = mean(policy turnaround / FCFS turnaround)
      - deadline set:   score = mean deadline miss rate (EDF shown for reference)
    Checkpoint score = turnaround ratio + deadline miss rate (lower is better),
    so a 10-point miss-rate drop is worth a 10% turnaround change. The best
    checkpoint is saved and every evaluation appended to the CSV learning curve.
    """

    def __init__(self, mode, num_cores, save_path, log_path, eval_freq, lstm=None):
        super().__init__()
        self.mode, self.num_cores = mode, num_cores
        self.save_path, self.log_path = save_path, log_path
        self.eval_freq, self.lstm = eval_freq, lstm
        self.workloads = validation_workloads(num_cores)
        self.fcfs = [_traditional(FCFSScheduler(), w, num_cores)["avg_turnaround_time"] for w in self.workloads]
        self.dl_workloads = deadline_validation_workloads(num_cores)
        self.edf_miss = float(np.mean([_traditional(EDFScheduler(), w, num_cores)["deadline_miss_rate"]
                                       for w in self.dl_workloads]))
        self.best = float("inf")
        self.next_eval = eval_freq
        self.start = time.time()
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["timesteps", "minutes", "ep_rew_mean", "val_ratio_fcfs",
                                    "val_miss_rate", "edf_miss_rate", "score", "best"])

    def evaluate(self):
        ratios = [run_policy(self.model, self.mode, w, self.num_cores, self.lstm)["avg_turnaround_time"] / f
                  for w, f in zip(self.workloads, self.fcfs)]
        misses = [run_policy(self.model, self.mode, w, self.num_cores, self.lstm)["deadline_miss_rate"]
                  for w in self.dl_workloads]
        return float(np.mean(ratios)), float(np.mean(misses))

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval += self.eval_freq
        ratio, miss = self.evaluate()
        score = ratio + miss
        improved = score < self.best
        if improved:
            self.best = score
            self.model.save(self.save_path)
        ep_rew = [e["r"] for e in self.model.ep_info_buffer] if self.model.ep_info_buffer else [float("nan")]
        minutes = (time.time() - self.start) / 60
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([self.num_timesteps, f"{minutes:.1f}", f"{np.mean(ep_rew):.2f}",
                                    f"{ratio:.4f}", f"{miss:.4f}", f"{self.edf_miss:.4f}", f"{score:.4f}", int(improved)])
        print(f"[{self.mode}-{self.num_cores}c] {self.num_timesteps:>9,} steps {minutes:6.1f} min | "
              f"turnaround/FCFS {ratio:.4f}  miss {miss:.3f} (EDF {self.edf_miss:.3f})  score {score:.4f}"
              f"{'  * saved' if improved else ''}", flush=True)
        return True


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

DEFAULT_HPARAMS = dict(gamma=0.995, rollout=4096, batch=512, lr=3e-4, net=128, epochs=10, log_std=-1.5)
# Chosen by a 2-core sweep (gamma, batch size, exploration noise, learning rate):
# low initial exploration noise (std ~0.22 on the weights) matters most; the
# Hybrid additionally prefers a lower learning rate.
MODE_HPARAMS = {"ppo": {}, "hybrid": {"lr": 1e-4}}


def train(mode: str, num_cores: int, timesteps: int, n_envs: int, seed: int, eval_freq: int,
          out_dir: str = None, **hp):
    hp = {**DEFAULT_HPARAMS, **MODE_HPARAMS[mode], **{k: v for k, v in hp.items() if v is not None}}
    log_dir = os.path.join(out_dir, "training_logs") if out_dir else LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    torch.set_num_threads(2)

    venv_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    venv = venv_cls([_env_factory(mode, num_cores, i, seed) for i in range(n_envs)])
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=hp["gamma"])

    lr = hp["lr"]
    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=lambda progress: lr * progress,   # linear decay to 0
        n_steps=hp["rollout"] // n_envs,
        batch_size=hp["batch"],
        n_epochs=hp["epochs"],
        gamma=hp["gamma"],
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[hp["net"]] * 2, vf=[hp["net"]] * 2), log_std_init=hp["log_std"]),
        seed=seed,
        device="cpu",
        verbose=0,
    )

    lstm = load_lstm(LSTM_MODEL_PATH) if mode == "hybrid" else None
    save_path = output_path(mode, num_cores, out_dir)
    callback = ValidationCallback(mode, num_cores, save_path,
                                  os.path.join(log_dir, f"{mode}_{num_cores}c.csv"), eval_freq, lstm)
    print(f"\n=== Training {mode.upper()} on {num_cores} core(s): {timesteps:,} steps, {n_envs} envs, {hp} ===", flush=True)
    model.learn(total_timesteps=timesteps, callback=callback)
    # Final evaluation so the last policy is also a checkpoint candidate
    callback.next_eval = 0
    callback._on_step()
    venv.close()
    print(f"=== Done {mode}-{num_cores}c: best val score {callback.best:.4f} -> {save_path}.zip ===", flush=True)
    return callback.best


def main():
    parser = argparse.ArgumentParser(description="Train PPO / Hybrid CPU schedulers")
    parser.add_argument("--mode", choices=["ppo", "hybrid", "all"], default="all")
    parser.add_argument("--cores", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-freq", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Save models/logs here instead of models/ (for sweeps)")
    for name, typ in [("gamma", float), ("rollout", int), ("batch", int), ("lr", float), ("net", int), ("epochs", int), ("log_std", float)]:
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=typ, default=None, help=f"default {DEFAULT_HPARAMS[name]}")
    args = parser.parse_args()
    hp = {k: getattr(args, k) for k in DEFAULT_HPARAMS}

    if args.mode in ("hybrid", "all") and not os.path.exists(LSTM_MODEL_PATH):
        raise SystemExit(f"Hybrid training needs the LSTM: run python -m backend.ml.train_lstm first ({LSTM_MODEL_PATH})")

    modes = ["ppo", "hybrid"] if args.mode == "all" else [args.mode]
    for mode in modes:
        for cores in args.cores:
            train(mode, cores, args.timesteps, args.n_envs, args.seed, args.eval_freq, args.out_dir, **hp)


if __name__ == "__main__":
    main()
