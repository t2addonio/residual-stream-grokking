#!/usr/bin/env python3
import subprocess, os
from pathlib import Path

N_GPUS = 8
SEEDS = list(range(8))
WORKER = "worker_joint4.py"
OUT_ROOT = Path("gpt2_joint4_results")
LOG_DIR = Path("gpt2_joint4_logs")
OUT_ROOT.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Path to the previous multi-direction results that contain dirs.npy
PREV_ROOT = Path("gpt2_multi_results")

print("Launching simultaneous 4-direction cancellation...")
procs = []
for gpu_id, seed in enumerate(SEEDS):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_file = LOG_DIR / f"gpu{gpu_id}.log"
    prev_dir = PREV_ROOT / f"gpu{gpu_id}" / f"seed_{seed:03d}"
    cmd = [
        "python3", "-u", WORKER,
        "--seed", str(seed),
        "--prev_dir", str(prev_dir),
        "--out_dir", str(OUT_ROOT / f"gpu{gpu_id}"),
        "--device", "cuda:0",
    ]
    print(f"GPU {gpu_id} → seed {seed}")
    with open(log_file, "w") as f:
        procs.append(subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT))

for p in procs:
    p.wait()
print("All finished.")
