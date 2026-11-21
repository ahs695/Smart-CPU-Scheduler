# backend/paths.py
"""
Filesystem locations resolved from the repository root, so every entry point
(Flask app, training, experiments) works regardless of the current directory.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.pt")


def model_path(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)


def results_path(filename: str) -> str:
    return os.path.join(RESULTS_DIR, filename)


def ppo_model_candidates(mode: str, num_cores: int) -> list:
    """
    Model file(s) to try for a PPO/Hybrid policy at a core count:
    ppo_scheduler_{N}c.zip / ppo_scheduler_hybrid_{N}c.zip.
    (Models from before the Nov 2026 retrain live in models/archive_* and are
    incompatible with the current environment, so there is no legacy fallback.)
    """
    mode = mode.lower()
    base = "ppo_scheduler" if mode == "ppo" else f"ppo_scheduler_{mode}"
    return [model_path(f"{base}_{num_cores}c.zip")]
