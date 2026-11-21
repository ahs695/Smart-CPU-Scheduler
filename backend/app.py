# backend/app.py

import os
import copy
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from backend.simulator.process import Process
from backend.simulator.multi_core_simulator import MultiCoreSimulator
from backend.simulator.metrics import MetricsEngine
from backend.simulator.workload_generator import WorkloadGenerator

# Traditional Schedulers
from backend.simulator.traditional.fcfs import FCFSScheduler
from backend.simulator.traditional.sjf import SJFScheduler
from backend.simulator.traditional.round_robin import RoundRobinScheduler
from backend.simulator.traditional.mlfq import MLFQScheduler

# RL / Hybrid
from stable_baselines3 import PPO
from backend.rl.env import SchedulingEnv
from backend.hybrid.hybrid_scheduler import HybridSchedulingEnv
from backend.ml.lstm_model import MAX_BURST, load_lstm
from backend.hybrid.hybrid_scheduler import FALLBACK_BURST
from backend.ml.dataset import DatasetBuilder
from backend.paths import LSTM_MODEL_PATH, ppo_model_candidates

app = Flask(__name__)
CORS(app)

# --- Global Model Cache ---
# Models keyed by (mode, num_cores), e.g. ("ppo", 2), ("hybrid", 4)
MODELS = {}
LSTM_MODEL = None

SUPPORTED_CORES = [1, 2, 4, 8]

def _load_lstm():
    """Load the shared LSTM burst predictor into LSTM_MODEL (no-op if loaded or missing)."""
    global LSTM_MODEL
    if LSTM_MODEL is not None:
        return
    try:
        LSTM_MODEL = load_lstm(LSTM_MODEL_PATH)
    except Exception as e:
        print(f"Error loading LSTM: {e}")

def _load_policy(mode, num_cores):
    """Load and cache the PPO/Hybrid policy for a core count. Returns None if unavailable."""
    key = (mode, num_cores)
    if key not in MODELS:
        for path in ppo_model_candidates(mode, num_cores):
            if os.path.exists(path):
                try:
                    MODELS[key] = PPO.load(path)
                    break
                except Exception as e:
                    print(f"Error loading {path}: {e}")
    return MODELS.get(key)

def load_models():
    _load_lstm()
    for nc in SUPPORTED_CORES:
        _load_policy("ppo", nc)
        _load_policy("hybrid", nc)

# Initial load
load_models()

# --- Helper Logic ---

TRADITIONAL_SCHEDULERS = ["FCFS", "SJF", "RR", "MLFQ"]

def get_scheduler(name, quantum=2):
    """Single source of truth for traditional schedulers, shared by
    /simulate and /compare so both endpoints run identical algorithms."""
    name = name.upper()
    if name == "FCFS": return FCFSScheduler()
    if name == "SJF": return SJFScheduler(preemptive=False)
    if name == "RR": return RoundRobinScheduler(quantum=quantum)
    if name == "MLFQ": return MLFQScheduler()
    return None

def _parse_quantum(data):
    """Returns (quantum, error_message)."""
    try:
        quantum = int(data.get("quantum", 2))
    except (TypeError, ValueError):
        return None, "quantum must be an integer"
    if quantum < 1:
        return None, "quantum must be >= 1"
    return quantum, None

def run_rl_sim(mode, processes, num_cores):
    num_cores = int(num_cores)

    model = _load_policy(mode.lower(), num_cores)
    if model is None:
        return None

    if mode.lower() == "hybrid":
        _load_lstm()

    if mode.lower() == "hybrid":
        env = HybridSchedulingEnv(
            processes=copy.deepcopy(processes),
            num_cores=num_cores,
            lstm_model=LSTM_MODEL
        )
    else:
        env = SchedulingEnv(
            processes=copy.deepcopy(processes),
            num_cores=num_cores
        )
    
    state, _ = env.reset()
    trace = []
    
    while True:
        tick_time = env.simulator.time
        action, _ = model.predict(state, deterministic=True)
        state, _, terminated, truncated, _ = env.step(action)
        
        # Capture trace AFTER stepping — use gantt_chart for the PID that
        # actually ran during this tick (avoids phantom idle after completion)
        sim = env.simulator
        current_state = {
            "time": tick_time,
            "ready_queue": [p.pid for p in sim.ready_queue],
            "cores": [
                {"id": c.core_id, "pid": sim.gantt_chart[c.core_id][-1] if sim.gantt_chart[c.core_id] else None}
                for c in sim.cores
            ]
        }
        trace.append(current_state)
        
        if terminated or truncated:
            break
            
    sim = env.simulator
    metrics = MetricsEngine.summarize(sim.processes, sim.cores, sim.time)
    
    return {
        "gantt_chart": sim.gantt_chart,
        "metrics": metrics,
        "trace": trace
    }

def run_traditional_sim(scheduler, processes, num_cores):
    sim = MultiCoreSimulator(
        processes=copy.deepcopy(processes),
        scheduler=scheduler,
        num_cores=num_cores
    )
    
    trace = []
    sim.reset()
    while not sim._all_completed() and sim.time < 5000: # Safety cap
        tick_time = sim.time
        sim.step()
        # Capture trace AFTER step() — use gantt_chart for the PID that
        # actually ran during this tick (avoids phantom idle after completion)
        current_state = {
            "time": tick_time,
            "ready_queue": [p.pid for p in sim.ready_queue],
            "cores": [
                {"id": c.core_id, "pid": sim.gantt_chart[c.core_id][-1] if sim.gantt_chart[c.core_id] else None}
                for c in sim.cores
            ]
        }
        trace.append(current_state)
        
    metrics = MetricsEngine.summarize(sim.processes, sim.cores, sim.time)
    
    return {
        "gantt_chart": sim.gantt_chart,
        "metrics": metrics,
        "trace": trace
    }

# --- Routes ---

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "lstm_loaded": LSTM_MODEL is not None,
        "models_loaded": {f"{mode}_{nc}c": True for (mode, nc) in MODELS}
    })

@app.route("/simulate", methods=["POST"])
def simulate():
    data = request.json or {}
    sched_name = data.get("scheduler", "FCFS")
    num_cores = int(data.get("num_cores", 2))
    process_data = data.get("processes", [])
    quantum, quantum_error = _parse_quantum(data)
    if quantum_error:
        return jsonify({"error": quantum_error}), 400

    processes, process_error = _parse_processes(process_data)
    if process_error:
        return jsonify({"error": process_error}), 400

    if not processes:
        # Fallback to a small random workload if none provided
        processes = WorkloadGenerator.mixed(10)
        
    if sched_name in ["PPO", "HYBRID"]:
        result = run_rl_sim(sched_name, processes, num_cores)
        if result is None:
            return jsonify({"error": f"No trained {sched_name} model for {num_cores}-core configuration. Train a model first."}), 400
    else:
        scheduler = get_scheduler(sched_name, quantum=quantum)
        if not scheduler:
            return jsonify({"error": "Unknown scheduler"}), 400
        result = run_traditional_sim(scheduler, processes, num_cores)
        
    if not result:
        return jsonify({"error": "Simulation failed (check if models are loaded)"}), 500

    return jsonify(result)

def _parse_processes(process_data):
    """
    Convert request process dicts to Process objects. Returns (processes, error_message).
    Rejects bursts < 1 and negative arrivals: such processes never complete, so the
    simulation would spin until its safety cap.
    """
    processes = []
    for i, p in enumerate(process_data or []):
        try:
            pid = int(p.get("pid", i))
            arrival = int(p.get("arrival", p.get("arrival_time", 0)))
            burst = int(p.get("burst", p.get("burst_time", 5)))
            priority = int(p.get("priority", 0))
        except (TypeError, ValueError, AttributeError):
            return None, f"process {i}: pid, arrival, burst and priority must be integers"
        # Optional program / executable label (any string or number); blank = unknown
        # (normalised to a string so 3 and "3" are the same program)
        program = p.get("program")
        if program is not None and not isinstance(program, (str, int)):
            return None, f"process P{pid}: program must be a string or integer"
        program = str(program).strip() if program is not None else ""
        program = program or None
        if burst < 1:
            return None, f"process P{pid}: burst must be >= 1"
        if arrival < 0:
            return None, f"process P{pid}: arrival must be >= 0"
        processes.append(Process(pid=pid, arrival_time=arrival, burst_time=burst, priority=priority, program=program))
    return processes, None

def _safe_metrics(runner):
    """Runs a scheduler, returns metrics or an error record. Never raises."""
    try:
        result = runner()
        if result is None:
            return {"error": "model_unavailable"}
        return {"metrics": result["metrics"], "total_time": result["metrics"]["total_time"]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

@app.route("/compare", methods=["GET", "POST"])
def compare():
    """
    Run every scheduler on the same input and return live metrics.

    POST body (preferred):
        { "processes": [...], "num_cores": 2, "quantum": 2 }

    If called with GET or with no processes, a small mixed workload is
    generated so the dashboard stays functional.
    """
    data = request.get_json(silent=True) or {}
    num_cores = int(data.get("num_cores", 2))
    quantum, quantum_error = _parse_quantum(data)
    if quantum_error:
        return jsonify({"error": quantum_error}), 400

    processes, process_error = _parse_processes(data.get("processes"))
    if process_error:
        return jsonify({"error": process_error}), 400
    if not processes:
        processes = WorkloadGenerator.mixed(10)

    # Traditional schedulers come from get_scheduler() — same config as /simulate.
    schedulers = {
        name: (lambda n=name: run_traditional_sim(get_scheduler(n, quantum=quantum), processes, num_cores))
        for name in TRADITIONAL_SCHEDULERS
    }
    schedulers["PPO"] = lambda: run_rl_sim("ppo", processes, num_cores)
    schedulers["Hybrid"] = lambda: run_rl_sim("hybrid", processes, num_cores)

    comparison = {name: _safe_metrics(runner) for name, runner in schedulers.items()}

    return jsonify({
        "comparison": comparison,
        "input": {
            "num_cores": num_cores,
            "num_processes": len(processes),
            "total_burst": sum(p.burst_time for p in processes),
        },
    })

LSTM_EVAL_SAMPLES = 50      # points plotted
LSTM_EVAL_WORKLOADS = 200   # held-out workloads the MAE is computed on
LSTM_EVAL_SEED = 42
_lstm_eval_cache = None

@app.route("/lstm-predictions", methods=["GET"])
def lstm_predictions():
    """
    Real LSTM predictions vs ground truth on seeded, held-out workloads built
    exactly like its training data (same-program job histories; see
    backend/ml/dataset.py). Values are in burst-time ticks. The baseline
    predicts the program's previous finished burst.
    """
    global _lstm_eval_cache
    _load_lstm()
    if LSTM_MODEL is None:
        return jsonify({"error": "LSTM model not available"}), 503

    if _lstm_eval_cache is None:
        groups = DatasetBuilder.generate(LSTM_EVAL_WORKLOADS, seed=LSTM_EVAL_SEED)
        X = [x for g in groups.values() for x in g[0]]
        y = [t * MAX_BURST for g in groups.values() for t in g[1]]
        hist = [h for g in groups.values() for h in g[2]]
        with torch.no_grad():
            preds = (LSTM_MODEL(torch.tensor(X, dtype=torch.float32)).clamp(0, 1) * MAX_BURST).tolist()
        baseline = [h[-1] if h else FALLBACK_BURST for h in hist]

        n = len(y)
        step = max(1, n // LSTM_EVAL_SAMPLES)
        idx = list(range(0, n, step))[:LSTM_EVAL_SAMPLES]
        _lstm_eval_cache = {
            "samples": [{"id": k, "actual": float(y[i]), "predicted": float(preds[i])} for k, i in enumerate(idx)],
            "mae": sum(abs(a - p) for a, p in zip(y, preds)) / n,
            "baseline_mae": sum(abs(a - b) for a, b in zip(y, baseline)) / n,
            "n": n,
        }

    return jsonify(_lstm_eval_cache)

if __name__ == "__main__":
    app.run(debug=True, port=5001)