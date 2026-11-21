# backend/hybrid/hybrid_scheduler.py
"""
Hybrid scheduler environment: the PPO priority function of SchedulingEnv plus
LSTM burst predictions as extra per-process features.

A process's burst is predicted the first time the scheduler sees it, from
the bursts of jobs of the *same program* that have already finished
(history_window in backend/ml/lstm_model.py — the input format the LSTM was
trained on). Only finished jobs' lengths are used, which a real OS knows; the
true burst of a running or waiting process is never seen. Processes without a
program label share one history (all finished jobs).
"""

from typing import List, Optional

import numpy as np
import torch

from backend.rl.env import SchedulingEnv, TIME_SCALE, LOG_SCALE
from backend.ml.lstm_model import MAX_BURST, SEQ_LENGTH, history_window

FALLBACK_BURST = 15.0  # prediction when there is no history and no LSTM


class HybridSchedulingEnv(SchedulingEnv):

    def __init__(
        self,
        processes=None,
        num_cores: int = 2,
        max_queue_size: int = 50,
        max_steps: int = 15000,
        lstm_model=None,
        use_predictions: bool = True,
        **kwargs,
    ):
        self.lstm_model = lstm_model
        self.use_predictions = use_predictions
        if use_predictions and lstm_model is None:
            print("⚠ HybridSchedulingEnv: no LSTM model given — falling back to window-mean predictions")
        self.process_predictions = {}
        super().__init__(
            processes=processes,
            num_cores=num_cores,
            max_queue_size=max_queue_size,
            max_steps=max_steps,
            **kwargs,
        )

    def _extra_feature_names(self) -> List[str]:
        return ["pred_burst", "pred_remaining", "log_pred_remaining", "pred_history", "pred_laxity"]

    def _on_reset(self):
        self.process_predictions = {}     # id(process) -> (predicted burst, history length)
        self.program_history = {}         # program -> finished bursts, in completion order
        self._completed_seen = 0

    def _sync_history(self):
        completed = self.simulator.completed_processes
        for p in completed[self._completed_seen:]:
            self.program_history.setdefault(p.program, []).append(p.burst_time)
        self._completed_seen = len(completed)

    def _predict(self, procs):
        """Predict bursts for processes seen for the first time (batched)."""
        new = [p for p in procs if id(p) not in self.process_predictions]
        if not new:
            return
        self._sync_history()
        histories = [self.program_history.get(p.program, []) for p in new]

        if not self.use_predictions:
            preds = [0.0] * len(new)
        elif self.lstm_model is not None:
            X = torch.tensor([history_window(h) for h in histories], dtype=torch.float32)
            with torch.no_grad():
                preds = (self.lstm_model(X).clamp(0.0, 1.0) * MAX_BURST).tolist()
        else:
            preds = [float(np.mean(h[-SEQ_LENGTH:])) if h else FALLBACK_BURST for h in histories]

        for p, pred, h in zip(new, preds, histories):
            self.process_predictions[id(p)] = (max(float(pred), 1.0), min(len(h), SEQ_LENGTH) / SEQ_LENGTH)

    def _extra_features(self, procs, attained: np.ndarray) -> Optional[np.ndarray]:
        self._predict(procs)
        pred = np.array([self.process_predictions[id(p)][0] for p in procs], dtype=np.float32)
        hist = np.array([self.process_predictions[id(p)][1] for p in procs], dtype=np.float32)
        pred_remaining = np.maximum(pred - attained, 0.0)
        # Predicted laxity: time to the deadline minus predicted remaining work (0 without a deadline)
        laxity = np.array([
            0.0 if p.deadline is None else (p.deadline - self._feature_time - r) / TIME_SCALE
            for p, r in zip(procs, pred_remaining)
        ], dtype=np.float32)
        return [
            pred / TIME_SCALE,
            pred_remaining / TIME_SCALE,
            np.log1p(pred_remaining) / LOG_SCALE,
            hist,
            np.clip(laxity, -2.0, 2.0),
        ]
