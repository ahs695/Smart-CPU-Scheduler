# AIMOS — Smart CPU Scheduler

A multi-core CPU scheduling simulator that compares classic scheduling
algorithms with **learned schedulers** (reinforcement learning, and RL plus an
LSTM burst predictor), with an interactive web dashboard.

- **Phase 1:** simulator, classic schedulers, PPO and Hybrid (PPO + LSTM)
  schedulers trained to minimise turnaround time, and the dashboard.
- **Phase 2:** real-time and enterprise scheduling: deadlines, periodic tasks,
  EDF and RMS, Google/Alibaba cluster-trace import, SLA breach alerts,
  deadline-aware retraining of PPO and Hybrid, and a light/dark theme.

More detail:

| File | What it covers |
|---|---|
| [`PHASE2.txt`](PHASE2.txt) | Everything done in Phase 2, file by file, with results |
| [`DASHBOARD_GUIDE.txt`](DASHBOARD_GUIDE.txt) | Which dashboard options to pick to check results yourself, plus showcase demos |
| [`BUGFIX_ANALYSIS.md`](BUGFIX_ANALYSIS.md) | Bugs found and fixed, and why the models were retrained |

---

## Contents

1. [How it works](#1-how-it-works)
2. [Schedulers](#2-schedulers)
3. [The learned schedulers (PPO and Hybrid)](#3-the-learned-schedulers-ppo-and-hybrid)
4. [Real-time scheduling (Phase 2)](#4-real-time-scheduling-phase-2)
5. [Metrics](#5-metrics)
6. [Results: Phase 1 vs Phase 2](#6-results-phase-1-vs-phase-2)
7. [Running the app](#7-running-the-app)
8. [Using the dashboard](#8-using-the-dashboard)
9. [Experiments and training](#9-experiments-and-training)
10. [API](#10-api)
11. [Project structure](#11-project-structure)
12. [Limitations](#12-limitations)

---

## 1. How it works

Time advances in discrete **ticks**. A workload is a list of **processes**
(jobs), each with:

| Field | Meaning |
|---|---|
| `pid` | Process id |
| `arrival` | Tick at which the job enters the system |
| `burst` | CPU ticks the job needs (1–50) |
| `priority` | Static priority (used for display; schedulers here do not rely on it) |
| `program` | Label of the program that produced the job (e.g. `build`, `shell`). The LSTM learns burst patterns per program. |
| `deadline` *(Phase 2, optional)* | Absolute tick by which the job must finish |
| `period` *(Phase 2, optional)* | Set only on jobs of a periodic real-time task; shown as `T=x` |

On every tick the simulator (`backend/simulator/multi_core_simulator.py`):

1. Moves newly arrived jobs into the ready queue.
2. Asks the scheduler which job each core should run.
3. Runs every core for one tick and records the Gantt chart.
4. Collects finished jobs.

The same simulator runs every scheduler, so results are directly comparable.

## 2. Schedulers

| Scheduler | Type | How it picks | Knows true burst? |
|---|---|---|---|
| **FCFS** | Non-preemptive | Earliest arrival first | No |
| **SJF** | Non-preemptive (dashboard) | Shortest burst first | **Yes (oracle)** |
| **SRTF** | Preemptive | Shortest remaining time first (experiments only) | **Yes (oracle)** |
| **RR** | Preemptive | Each job runs for a time quantum, then goes to the back | No |
| **MLFQ** | Preemptive | Multiple queues; jobs that use a lot of CPU drop to lower priority | No |
| **EDF** *(Phase 2)* | Preemptive | Earliest deadline first; jobs without a deadline go last | No |
| **RMS** *(Phase 2)* | Preemptive, fixed priority | Shortest period first; jobs without a period go last (then it acts like FCFS) | No |
| **PPO** | Learned | Learned priority function (section 3) | No |
| **Hybrid** | Learned | PPO + LSTM predicted burst | No (predicts it) |

SJF and SRTF are **oracles**: a real OS cannot know how long a job will run.
They are reference points, not deployable schedulers.

## 3. The learned schedulers (PPO and Hybrid)

**Environment** (`backend/rl/env.py`, Gymnasium). The agent learns a
*priority function*. Every tick it outputs one weight per job feature; each
runnable job is scored as `weights · features` and the highest-scoring jobs
run (one per core). Jobs that stay selected keep their core, so there are no
needless switches.

The agent only sees what a real OS could see: waiting time, CPU time received,
age, whether the job is running, and (Phase 2) deadline features: has a
deadline, time left, urgency, overdue. It never sees the true burst.

**Hybrid** (`backend/hybrid/hybrid_scheduler.py`) adds an **LSTM burst
predictor** (`backend/ml/`). When a job first appears, the LSTM predicts its
burst from the bursts of finished jobs of the *same program*. The Hybrid also
gets `pred_laxity` = time left before the deadline − predicted remaining work.
Held-out LSTM error (MAE): 5.79 ticks, vs 6.47 for "repeat last burst" and
5.91 for a moving average.

**Reward** (`backend/rl/reward.py`), per tick:

```
r = -(jobs in system) / 10          # sums to total turnaround (Little's law)
    - 0.05 × context switches
    - 1.0  × deadlines missed this tick        (Phase 2)
    - 0.1  × jobs already past their deadline  (Phase 2, = lateness)
```

**Training** (`backend/rl/train_ppo.py`, Stable-Baselines3 PPO): 8 models,
PPO and Hybrid for 1, 2, 4 and 8 cores, 5M steps each. Each episode uses a
fresh random workload: 15% periodic task sets, about 38% aperiodic with
deadlines, about 47% without deadlines. The checkpoint with the best
validation score (turnaround vs FCFS + deadline miss rate) is kept.

Models live in `models/` (shipped as `models.zip`). The Phase 1
(turnaround-only) models are archived in
`models/archive_2026-11-21_turnaround_only/`.

## 4. Real-time scheduling (Phase 2)

**Deadlines.** A job *meets* its deadline if it completes at or before it;
otherwise it is a *miss*, and *lateness* = completion − deadline. Deadlines
come from:

- **Periodic tasks** (RT presets): a task with period `T` and execution time
  `C` releases a new job every `T` ticks. Each release is a separate job with
  `deadline = release + T` (it must finish before the next one). The dashboard
  expands each task over two hyperperiods, so the same task appears many times
  in the table and the Gantt chart.
- **Auto Deadlines**: `deadline = arrival + 3 × burst` for every job without one.
- **Traces**: Google jobs get a deadline from their scheduling class (class 3:
  2× burst, class 2: 3×, class 1: 5×, class 0: none). Alibaba jobs get none
  (use Auto Deadlines).
- **Typing them in** the Deadline column.

**Schedulability.** Utilisation `U = Σ C/T`. On one core:

- **EDF** meets every deadline whenever `U ≤ 1` (it is optimal).
- **RMS** is guaranteed only when `U ≤ n(2^(1/n) − 1)`, the Liu & Layland
  bound (0.78 for 3 tasks).

The presets show this:

| Preset | Tasks (T, C) | U | Result on 1 core |
|---|---|---|---|
| RT Schedulable | sensor (8,2), control (12,3), logger (16,4) | 0.75 | EDF 0/26 misses, RMS 0/26 |
| RT Overloaded | sensor (6,2), control (8,3), logger (12,3) | 0.96 | EDF 0/18, RMS 2/18 (logger at t=12 and t=36) |

**Cluster traces** (`backend/experiments/trace_import.py`). The importer
reads the Google 2011 `task_events` and Alibaba 2018 `batch_task` CSV formats
(auto-detected by column count). It takes 60 jobs and rescales time so the
95th-percentile duration becomes 50 ticks, clipping bursts to [1, 50].
Synthetic sample files are in `backend/experiments/sample_traces/`; real
traces come from Google's `clusterdata` and Alibaba's `clusterdata` GitHub
repositories.

**SLA Breach Alerts.** While the Gantt chart plays, the dashboard lists each
deadline miss at the tick it happens, with a running counter.

## 5. Metrics

| Metric | Definition | Better |
|---|---|---|
| Avg waiting time | Time in the ready queue = turnaround − burst | Lower |
| Avg turnaround time | Completion − arrival | Lower |
| Avg response time | First run − arrival | Lower |
| Throughput | Jobs completed per tick | Higher |
| CPU utilisation | Share of core-ticks spent running a job | Higher |
| Context switches | Times a core switched to a different job | Lower |
| Fairness index | Jain's index over each job's rate `burst / turnaround`: `(Σx)² / (n·Σx²)`. 1 = every job slowed by the same proportion, 1/n = one job got everything | Higher |
| Deadline miss rate *(Phase 2)* | Misses / jobs with a deadline (unfinished counts as a miss) | Lower |
| Avg / max lateness *(Phase 2)* | Ticks past the deadline, over jobs with a deadline | Lower |

Fairness and speed usually trade off: RR and MLFQ are the fairest (≈0.85) but
about 20% slower than FCFS.

## 6. Results: Phase 1 vs Phase 2

Phase 1 = turnaround-only models. Phase 2 = deadline-aware models. Data is in
`results/` and the archived Phase 1 results.

**Deadlines** (average over 21 settings from `run_realtime.py`; lower is better):

| Metric | EDF | FCFS | PPO P1 → P2 | Hybrid P1 → P2 |
|---|---|---|---|---|
| Miss rate, all settings | 20.9% | 32.4% | 30.6 → **22.7%** | 26.7 → **22.3%** |
| Miss rate, aperiodic | 46.3% | 64.0% | 62.4 → **49.0%** | 56.1 → **45.6%** |
| Miss rate, periodic | 1.9% | 8.8% | 6.7 → **3.0%** | 4.6 → 4.8% |
| Avg lateness, aperiodic (ticks) | 58.6 | 91.8 | 93.9 → **59.0** | 86.2 → **56.9** |
| Settings (of 21) tying or beating EDF | – | – | 5 → **7** | 6 → **11** |

**Speed without deadlines** (200 runs: 5 workload types × 1/2/4/8 cores × 10 seeds):

| Metric | FCFS | RR | MLFQ | SJF (oracle) | PPO P1 → P2 | Hybrid P1 → P2 |
|---|---|---|---|---|---|---|
| Turnaround / FCFS | 1.00 | 1.20 | 1.22 | 0.80 | 1.01 → **1.01** | 0.96 → **0.96** |
| Avg turnaround (ticks) | 120.6 | 164.7 | 165.6 | 91.0 | 123.0 → **121.0** | 117.3 → **115.3** |
| Context switches / run | 0 | 240 | 184 | 0 | 95.1 → **30.2** | 31.0 → **9.2** |
| Fairness index | 0.63 | 0.84 | 0.85 | 0.75 | 0.63 → 0.64 | 0.66 → **0.67** |

**Summary.** Phase 2 made the learned schedulers deadline-aware without losing
Phase 1 speed:

- Deadline misses fell by 26% (PPO) and 16% (Hybrid).
- Lateness fell by about 35%.
- Context switches fell by about 70%.
- On aperiodic workloads the Hybrid now beats EDF on both miss rate and
  lateness.
- EDF still has the best overall miss rate, and wins on fully loaded
  single-core periodic sets.

## 7. Running the app

Requirements: **Python 3.12–3.14** and **Node.js 20+**. The backend runs on
port **5001**, the frontend on **5173**.

### macOS / Linux

```bash
# one-time setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
unzip -o models.zip               # creates models/ (not tracked in git)
cd frontend && npm install && cd ..

# terminal 1 — backend
source .venv/bin/activate
python -m backend.app

# terminal 2 — frontend
cd frontend && npm run dev
```

### Windows (PowerShell)

```powershell
# one-time setup
python -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # only if activation is blocked
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Expand-Archive -Path models.zip -DestinationPath . -Force
cd frontend; npm install; cd ..

# terminal 1 — backend
.\.venv\Scripts\Activate.ps1
python -m backend.app

# terminal 2 — frontend
cd frontend
npm run dev
```

In cmd, activate with `.venv\Scripts\activate.bat` and unzip with
`tar -xf models.zip`.

Open **http://localhost:5173**. The *Backend* pill at the top turns green
once the API is reachable.

**Troubleshooting**

- **`No module named 'flask'` / `'numpy'`:** the install stopped part-way.
  Re-run `python -m pip install -r requirements.txt` with the venv active and
  read the first `ERROR:` line. If a package has no wheel for your Python
  version, recreate the venv with Python 3.12 or 3.13.
- **On Windows, `python3`** is a different, global Python. Use `python`
  inside the venv.
- **PPO/Hybrid show an error:** `models.zip` was not extracted into the
  project root. You should have `models/ppo_scheduler_1c.zip`.
- **Run the backend from the project root**, not from inside `backend/`.

## 8. Using the dashboard

**Left panel**

| Control | Effect |
|---|---|
| Scheduler | Which algorithm the Gantt chart, metric cards and SLA alerts show |
| CPU Cores | 1, 2, 4 or 8 (default 2); also used by the comparison table |
| RR Time Quantum | Only affects Round Robin |
| Mixed / Compute Heavy / IO Intensive | Small aperiodic workloads (no deadlines until you add them) |
| RT Schedulable / RT Overloaded | Periodic real-time task sets |
| Auto Deadlines | `deadline = arrival + 3 × burst` for jobs without one |
| Import Cluster Trace | Load 60 jobs from a Google/Alibaba trace (format: Auto-detect, Google or Alibaba) |
| Workload table | Edit arrival, burst, priority, program and deadline; the comparison re-runs automatically |

**Right side**

| Panel | Shows |
|---|---|
| Gantt chart | Per-core timeline; Play to animate, scroll-wheel to zoom; jobs that missed are outlined red |
| SLA Breach Alerts | Misses of the selected scheduler as playback reaches them |
| Metric cards | Selected scheduler's metrics (deadline cards only when deadlines exist) |
| Comparison table / chart | **Every** scheduler on the same workload, including a Deadline Miss column |
| Training charts | LSTM predicted vs actual bursts (with MAE) |

**Quick demos** (2 cores, preset → Auto Deadlines → Run; deadline-miss %):

| Demo | PPO | Hybrid | EDF | SJF | FCFS | RR |
|---|---|---|---|---|---|---|
| Mixed | **0** | **10** | 20 | 30 | 40 | 70 |
| IO Intensive | **0** | **0** | 9.1 | 9.1 | 45.5 | 45.5 |
| Compute Heavy | **0** | **0** | 14.3 | 0 | 42.9 | 42.9 |

On 1 core with RT Overloaded, select RMS and then EDF to see RMS miss and EDF
not miss. See `DASHBOARD_GUIDE.txt` for more recipes and the expected numbers.

## 9. Experiments and training

Run from the project root with the venv active:

```bash
python -m backend.experiments.run_baselines        # classic schedulers -> results/baselines.csv
python -m backend.experiments.run_rl               # PPO + Hybrid       -> results/rl_results.csv
python -m backend.experiments.run_realtime         # deadline suites    -> results/realtime.csv
python -m backend.experiments.statistical_tests    # t-tests            -> results/summary.csv
python -m backend.experiments.plot_results         # charts             -> results/plots/

python -m backend.ml.train_lstm                    # retrain LSTM predictor
python -m backend.rl.train_ppo --mode all --cores 1 2 4 8   # retrain PPO + Hybrid (5M steps each)
```

Frontend checks: `cd frontend && npx eslint src && npm run build`.

## 10. API

Flask app in `backend/app.py`, at `http://localhost:5001`.

| Method & route | Purpose |
|---|---|
| `GET /` | Health check |
| `POST /simulate` | Run one scheduler: `{processes, scheduler, num_cores, quantum}` → Gantt, trace, metrics, `deadline_events`, `realtime` block |
| `GET/POST /compare` | Run every scheduler on the same workload → per-scheduler metrics |
| `POST /workload/periodic` | Expand `{tasks:[{name, period, wcet}], …}` into jobs, plus utilisation and RMS bound |
| `POST /workload/deadlines` | Add deadlines (`arrival + ⌈burst × slack⌉`) |
| `POST /import-trace` | Multipart `file` + `format` (`auto`/`google`/`alibaba`), `max_jobs`, `slack` → processes |
| `GET /lstm-predictions` | LSTM predicted vs actual bursts for the training chart |

Invalid input returns HTTP 400 with a message.

## 11. Project structure

```
backend/
  app.py                     Flask API
  paths.py                   Model / results paths
  simulator/
    process.py               Process model (deadline, period)
    core.py                  One CPU core
    multi_core_simulator.py  Tick loop
    scheduler_base.py        Shared helpers (core-keeping assignment)
    traditional/             fcfs, sjf, round_robin, mlfq, edf, rms
    realtime.py              Periodic expansion, utilisation, Liu & Layland, deadlines
    metrics.py               All metrics incl. Jain's fairness and deadline metrics
    workload_generator.py    Seeded synthetic workloads
  rl/                        env.py, reward.py, train_ppo.py, evaluate_rl.py
  hybrid/                    Hybrid environment (PPO + LSTM features)
  ml/                        LSTM model, dataset, training, evaluation
  experiments/               run_* scripts, statistics, plots, trace_import.py,
                             sample_traces/ (synthetic Google + Alibaba files)
frontend/
  src/components/            Dashboard, InputPanel, GanttChart, MetricsPanel,
                             SLAAlerts, TrainingCharts
  src/lib/theme.js           Light/dark theme store and chart colours
  src/services/api.js        Backend client
models/                      Trained models (from models.zip)
results/                     Experiment CSVs
```

## 12. Limitations

- **EDF still has the lowest average miss rate** (20.9% vs Hybrid 22.3%). On
  fully loaded single-core periodic sets (U = 1.0) the models trail EDF
  (PPO 7.4%, Hybrid 21.3%, EDF 0%).
- **The models' advantage shows on busy multi-core and overloaded workloads.**
  On 1 core the small presets favour EDF.
- **Periodic jobs are independent in the simulation.** Repeated jobs of one
  task are separate processes, linked only by their shared period. The Gantt
  chart colours them by pid, not by task.
- **The simulator is a model, not an OS.** There is no I/O blocking, no
  context-switch cost in time, and integer ticks.
- **Trace import keeps only 60 rescaled jobs.** Real traces are not bundled.
