"""Orchestration: spawn/poll/save the Modal jobs that produce the result data.

Each subcommand spawns/polls/saves a class of jobs.

Usage:
    python3 orchestrate.py spawn_geometry           # 4 sizes x 52 ckpts = 208 jobs
    python3 orchestrate.py spawn_prep_tokens        # 1 job (small CPU)
    python3 orchestrate.py spawn_llc_calibration    # ~12 calib jobs
    python3 orchestrate.py spawn_llc_timeseries     # ~52 LLC + ~52 rLLC jobs
    python3 orchestrate.py poll <calls_file>        # poll calls in a json file
"""

import json
import os
import sys
import time
from pathlib import Path

import modal

APP = "project1-dow-development"

LOG_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
EARLY_STEPS = list(range(1000, 30001, 1000))
LATE_STEPS = list(range(40000, 140001, 10000)) + [143000]
ALL_STEPS = LOG_STEPS + EARLY_STEPS + LATE_STEPS

GEOM_SIZES = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
]

# subset of checkpoints used for the LLC time series (subsample to keep cost down)
LLC_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + \
            list(range(1000, 30001, 2000)) + \
            list(range(40000, 140001, 20000)) + [143000]

def save_calls(path: str, calls: dict):
    Path(path).write_text(json.dumps(calls, indent=2))
    print(f"saved {len(calls)} calls -> {path}")


def spawn_prep_tokens():
    fn = modal.Function.from_name(APP, "prepare_eval_tokens")
    fc = fn.spawn(seq_len=256, n_sequences=256)
    save_calls("data/calls_prep.json", {"prep_eval_tokens": fc.object_id})


def spawn_geometry():
    fn = modal.Function.from_name(APP, "geometry_at_checkpoint")
    calls = {}
    for size in GEOM_SIZES:
        for step in ALL_STEPS:
            fc = fn.spawn(size, step)
            calls[f"{size}|step{step}"] = fc.object_id
    save_calls("data/calls_geometry.json", calls)


def spawn_llc_calibration():
    """Calibrate on Pythia-160m mid-training: step 71000."""
    fn = modal.Function.from_name(APP, "llc_at_checkpoint")
    calls = {}
    grid = []
    for lr in [3e-5, 1e-4, 3e-4, 1e-3]:
        for loc in [10.0, 100.0, 1000.0]:
            grid.append({"lr": lr, "localization": loc})
    for cfg in grid:
        fc = fn.spawn(
            "EleutherAI/pythia-160m",
            71000,
            lr=cfg["lr"],
            localization=cfg["localization"],
            num_chains=4,
            num_draws=80,
            num_burnin_steps=40,
            batch_size=4,
            seq_len=256,
            n_sequences=256,
        )
        calls[f"calib_lr{cfg['lr']:.0e}_loc{cfg['localization']:g}"] = fc.object_id
    save_calls("data/calls_llc_calib.json", calls)


def spawn_llc_timeseries(lr: float, localization: float, layer_substring: str):
    """Once calibrated, run global LLC + rLLC across LLC_STEPS on Pythia-160m."""
    fn = modal.Function.from_name(APP, "llc_at_checkpoint")
    calls = {}
    for step in LLC_STEPS:
        fc = fn.spawn(
            "EleutherAI/pythia-160m",
            step,
            lr=lr,
            localization=localization,
            num_chains=4,
            num_draws=100,
            num_burnin_steps=50,
            batch_size=4,
            seq_len=256,
            n_sequences=256,
            layer_substring=None,
        )
        calls[f"global|step{step}"] = fc.object_id
        fc2 = fn.spawn(
            "EleutherAI/pythia-160m",
            step,
            lr=lr,
            localization=localization,
            num_chains=4,
            num_draws=100,
            num_burnin_steps=50,
            batch_size=4,
            seq_len=256,
            n_sequences=256,
            layer_substring=layer_substring,
        )
        calls[f"r{layer_substring}|step{step}"] = fc2.object_id
    save_calls("data/calls_llc_timeseries.json", calls)


def poll(calls_file: str, interval: int = 20):
    """Poll until all calls finish; save results into <calls_file>.results.json."""
    calls = json.loads(Path(calls_file).read_text())
    out_path = calls_file.replace("calls_", "results_")
    out: dict = {}
    if Path(out_path).exists():
        out = json.loads(Path(out_path).read_text())

    pending = {k: v for k, v in calls.items() if k not in out}
    print(f"{len(pending)} pending of {len(calls)} (already done: {len(out)})")
    while pending:
        for key, cid in list(pending.items()):
            fc = modal.FunctionCall.from_id(cid)
            try:
                res = fc.get(timeout=0)
                out[key] = res
                Path(out_path).write_text(json.dumps(out, indent=2))
                pending.pop(key)
                print(f"  [done] {key}", flush=True)
            except TimeoutError:
                continue
            except Exception as e:
                out[key] = {"error": f"{type(e).__name__}: {e}"}
                Path(out_path).write_text(json.dumps(out, indent=2))
                pending.pop(key)
                print(f"  [err]  {key}: {e}", flush=True)
        if pending:
            print(f"  ...{len(pending)} still running", flush=True)
            time.sleep(interval)
    print(f"\nAll done. Results: {out_path} ({len(out)} entries)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "spawn_prep_tokens":
        spawn_prep_tokens()
    elif cmd == "spawn_geometry":
        spawn_geometry()
    elif cmd == "spawn_llc_calibration":
        spawn_llc_calibration()
    elif cmd == "spawn_llc_timeseries":
        lr = float(sys.argv[2])
        loc = float(sys.argv[3])
        layer = sys.argv[4]
        spawn_llc_timeseries(lr, loc, layer)
    elif cmd == "poll":
        poll(sys.argv[2])
    else:
        print(__doc__)
