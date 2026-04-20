# Bugfix1 — Diagnosis and Fixes

Branch: `bugfix1`. Scope: simulator correctness, live scheduler comparison, RL/Hybrid failure-mode diagnosis.

---

## 1. Simulation Correctness

### Root Cause

Two historical bugs in `backend/simulator/multi_core_simulator.py`, both already patched in the current tree but recoverable from the git log. They cleanly explain all three observed symptoms:

**Bug A — Gantt log reads `core.current_process` *after* execute()**
Pre-`c2b5b6d` code:

```python
finished = core.execute(self.time, time_slice=1)
if core.current_process:                           # <-- read AFTER execute
    self.gantt_chart[core.core_id].append(core.current_process.pid)
else:
    self.gantt_chart[core.core_id].append(None)
if finished:
    completed.append(finished); self.completed_processes.append(finished)
```

`Core.execute()` sets `self.current_process = None` the moment the process completes (see [core.py:74-76](backend/simulator/core.py)). So on the tick a process *completes*, the Gantt branch sees `None` and logs `None`. On a *preempted* tick, the old process was already overwritten by `assign_process(new)` *before* execute, so `core.current_process` points at the new process → something gets logged, no visible gap.

**Bug B — duplicate residency in preemptive SJF**
Pre-`bbc66b3`, `SJFScheduler._preemptive` builds `runnable = ready_queue + [c.current_process for c in cores]` and sorts by `remaining_time`. If the same Process object appeared in both (ready_queue was stale after a race), it could be assigned to two cores. Both cores then call `Process.execute()`, and the second call sees `remaining_time == 0` and appends `0` to `execution_history`.

### Why It Manifests

| Symptom | Cause |
|---|---|
| Completed processes leave gaps in Gantt | Bug A — final tick of a completing process logs `None`. |
| Preempted don't leave gaps | Preemption rewrites `core.current_process` *before* execute, so the new pid is logged. |
| Total sim time "shorter than expected" | If the observer counts non-`None` Gantt cells as executed time, every completion loses one cell → perceived total shrinks by ~`num_processes` cells. `self.time` itself was always correct. |
| burst=1 records execution_time=0 | Bug B — same Process executed twice in one tick; second call records `min(0,1)=0`. |

### Fix (already in tree + defensive hardening added here)

- `c2b5b6d` inverted the log order: `if finished: log finished.pid elif current_process: log pid else: None`.
- `bbc66b3` added an `assigned_pids` dedup pass before `assign_process`.
- **New in this branch** ([core.py:30-60](backend/simulator/core.py)):
  - `Core.assign_process` now refuses a completed process (forces `None`).
  - `Core.execute` short-circuits to idle if it somehow sees a completed `current_process` instead of calling `Process.execute()` and appending a 0.
  - `_all_completed` changed from `>=` to `==` so a double-append surfaces loudly instead of masking.

### Further Improvement

Make `Process.execute` raise (or at minimum never append) on `remaining_time == 0`. Silent append-of-zero is exactly what let Bug B hide for a release cycle; a hard assertion would have caught it in the first simulation.

---

## 2. Real-Time Scheduler Comparison

### Root Cause

`/compare` in `app.py` returned a hard-coded summary regardless of input:

```python
summary = [{"name":"FCFS","waiting":45.2,...}, ...]
```

### Fix

Replaced with a live runner ([app.py:199-247](backend/app.py)):

- `POST /compare` accepts the same `processes`/`num_cores`/`quantum` shape as `/simulate`.
- `GET /compare` falls back to `WorkloadGenerator.mixed(10)` so the dashboard keeps working.
- Runs all six schedulers against an identical deep-copy of the input via the shared `run_traditional_sim` / `run_rl_sim` helpers.
- Each scheduler is wrapped in `_safe_metrics` so a missing PPO/Hybrid model becomes `{"error":"model_unavailable"}` instead of a 500.
- Response matches the requested shape: `{ "comparison": { "FCFS": {...}, ... }, "input": {...} }`.

Metric definitions are already consistent because every runner feeds `MetricsEngine.summarize(processes, cores, sim.time)`.

### Further Improvement

Add a `seed` field to the POST body and thread it through `WorkloadGenerator` so benchmark comparisons are reproducible. Right now the GET fallback is non-deterministic.

---

## 3. RL / Hybrid Performance Degradation

### Root Cause (diagnosis from the code, not generic RL advice)

**3a. Reward squashed through `tanh` destroys magnitude.**
`reward.py:87`: `normalized_reward = np.tanh(reward)`. Completions are weighted 5.0 and a single completion in a step already drives `tanh` into saturation (≈0.9999). A step with 3 completions is indistinguishable from a step with 1. The agent gets a binary signal, not a ranked one → gradients vanish → PPO drifts toward whatever its priors pick first, which looks like FCFS because the env sorts the pool by `(arrival_time, pid)` and action 0 is almost always the earliest arrival.

**3b. State is missing "which process is on which core".**
`env.py:_get_state` packs per-process features `(remaining, waiting, arrival)` sorted by arrival, plus a binary `core_busy` per core. The agent cannot tell which pool-index is currently residing on which core, so it cannot learn "keep current" vs "switch" — every action looks like a switch opportunity. This directly produces the excessive-context-switch symptom.

**3c. Action space overrides the agent.**
`env.py:203-210` has a "final sweep" that fills any core the agent left idle with the next unassigned process. The agent literally cannot emit an idle action; every training signal about idling is noise. Combined with 3b, PPO converges to "always keep indexing into the front of the arrival-sorted pool" — i.e. FCFS.

**3d. Hybrid adds the LSTM prediction to the state but the reward still penalizes "running a long job" at 0.005 per unit (`w_pred_burst`).**
After `tanh`, that term is zero. The LSTM feature has no gradient pressure behind it. Hybrid degrades relative to PPO when the extra state dimension adds variance without a reward signal to exploit it.

**3e. Context-switch term references `prev_running_burst` from the previous step.**
If the agent preempts to a predicted-shorter job, the *next* step's `delta_context` rewards it — but only the `w_context_good = 0.05` branch, already tanh-saturated. "Bad" switches cost 1.0 per switch (good). The asymmetry means the agent eventually learns: don't switch at all. Hence FCFS-like waiting.

### Fix

**State additions** (`env._get_state`):

- For each of the `num_cores` slots, emit a one-hot / index pointing to the pool position of the process currently on that core (or a sentinel for idle).
- Add per-process `on_core` bit (0/1) and `is_current` bit so the agent distinguishes "switch" from "keep".

**Action space:** drop the "final sweep" in `env.select_process`. Let the agent emit idle and pay for it via reward. The invalid-action penalty already exists; trust it.

**Reward, concrete replacement:**

```
r = w_c · Δcompleted
  − w_w · Δtotal_wait
  − w_s · (#switches_this_step)             # flat, not asymmetric
  − w_i · (idle_cores  if |ready|>0 else 0)
  − w_pb · Σ predicted_burst_on_core         # only if Hybrid
  + w_srtf · (1 if chose argmin(remaining) else 0)
```

Weights scaled so a typical step lands in roughly `[-3, +3]` and **the `tanh` is removed**. Use reward clipping `[-10, 10]` if stability matters — clipping preserves ordering, `tanh` does not.

**Hybrid specifically:** make the LSTM prediction drive reward, not just observation. Add `w_srtf_pred: +0.5` when the agent's chosen pool index coincides with `argmin(predicted_burst)`. Otherwise the LSTM channel is decorative.

### Further Improvement

Symmetric normalization: `remaining_time / max(pool_remaining)` instead of `/MAX_BURST=50.0`. Fixed denominators make short workloads collapse to near-zero features → the policy sees a constant vector → no gradient.

---

## 4. Architectural Note on Hybrid

Hybrid *shares* `SchedulingEnv.select_process` and only augments state + reward inputs. If PPO fails to exploit the extra LSTM channel, Hybrid has no mechanism to beat PPO — it's strictly noisier. To make the hierarchy meaningful, the Hybrid policy should either (a) hard-gate: fall back to SRTF on predicted-burst when PPO's value estimate is low-confidence, or (b) use the LSTM prediction as an *action prior* via a KL penalty during PPO training. Right now it's effectively "PPO with one extra number", which is why it sometimes regresses below PPO alone.
