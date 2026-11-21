# backend/experiments/sample_traces/make_synthetic.py
"""
Generate the SYNTHETIC sample traces in this folder (seeded, reproducible).
They only mimic the column layout of the Google 2011 task_events table and the
Alibaba 2018 batch_task.csv, for tests and dashboard demos. They contain no
real trace data. Run:  python -m backend.experiments.sample_traces.make_synthetic
"""
import csv
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))


def google(path, n_tasks=80, seed=1):
    rng = random.Random(seed)
    users = {f"user_{i}": rng.uniform(20, 400) for i in range(6)}   # typical runtime (s) per user
    rows, t = [], 600_000_000
    for i in range(n_tasks):
        t += int(rng.expovariate(1 / 15) * 1e6)
        user = rng.choice(list(users))
        job, task = 6_000_000 + i // 3, i % 3
        cls = rng.choice([0, 0, 1, 2, 3])
        wait = int(rng.uniform(0, 5) * 1e6)
        run = int(users[user] * rng.uniform(0.7, 1.3) * 1e6)
        base = ["", str(job), str(task), str(rng.randint(1, 500))]
        tail = [user, str(cls), str(rng.randint(0, 11)), f"{rng.uniform(0.01, 0.1):.4f}",
                f"{rng.uniform(0.01, 0.1):.4f}", "0", "0"]
        rows.append([str(t)] + base[:1] + base[1:3] + [""] + ["0"] + tail)            # SUBMIT
        rows.append([str(t + wait)] + base[:1] + base[1:4] + ["1"] + tail)            # SCHEDULE
        rows.append([str(t + wait + run)] + base[:1] + base[1:4] + ["4"] + tail)      # FINISH
    rows.sort(key=lambda r: int(r[0]))
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def alibaba(path, n_tasks=80, seed=2):
    rng = random.Random(seed)
    jobs = {f"j_{1000 + i}": rng.uniform(10, 300) for i in range(8)}
    rows, t = [], 86_400
    for i in range(n_tasks):
        t += int(rng.expovariate(1 / 8))
        job = rng.choice(list(jobs))
        dur = max(1, int(jobs[job] * rng.uniform(0.7, 1.3)))
        status = "Terminated" if rng.random() < 0.9 else "Failed"
        rows.append([f"M{rng.randint(1, 5)}", str(rng.randint(1, 50)), job, "1", status,
                     str(t), str(t + dur), "100", f"{rng.uniform(0.1, 1):.2f}"])
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


if __name__ == "__main__":
    google(os.path.join(HERE, "synthetic_google_task_events.csv"))
    alibaba(os.path.join(HERE, "synthetic_alibaba_batch_task.csv"))
    print("written to", HERE)
