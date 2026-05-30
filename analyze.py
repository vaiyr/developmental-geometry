"""Project 1 analysis: align manifold formation with LLC stage transitions.

Reads results_geometry.json and results_llc_timeseries.json, produces:
  - Per-size, per-step time series of best-layer order_score, resid/r, EVR
  - Pythia-160m: global LLC and rLLC(layers.7.) across training
  - Identification of t* (geometric formation point) and t** (LLC discontinuity)
  - Coincidence test: |t* - t**| <= 1 checkpoint?
"""

import json
from pathlib import Path

import numpy as np


def parse_geom(path: str = "data/results_geometry.json"):
    res = json.load(open(path))
    out = {}
    for k, v in res.items():
        if "error" in v:
            continue
        model, step_str = k.split("|step")
        size = model.split("/")[-1]
        step = int(step_str)
        if size not in out:
            out[size] = {}
        bp = v["best_layer_metrics"]
        out[size][step] = {
            "best_layer": v["best_layer"],
            "n_layers": v["n_layers"],
            "order_score": bp["order_score"],
            "resid_r": bp["circle_residual_ratio"],
            "evr_top2": sum(bp["pc_explained_var"]),
            "score": bp["score"],
            # also store the layer-7 metrics specifically for Pythia-160m alignment
            "per_layer": v["per_layer"],
        }
    for size in out:
        out[size] = dict(sorted(out[size].items()))
    return out


def parse_llc(path: str = "data/results_llc_timeseries.json"):
    res = json.load(open(path))
    global_, rllc = {}, {}
    for k, v in res.items():
        if "error" in v:
            continue
        prefix, step_str = k.split("|step")
        step = int(step_str)
        rec = {
            "llc_mean": v["llc_mean"],
            "llc_std": v["llc_std"],
            "init_loss": v["init_loss"],
            "active_params": v.get("active_params", None),
        }
        if prefix == "global":
            global_[step] = rec
        else:
            rllc[step] = rec
    return dict(sorted(global_.items())), dict(sorted(rllc.items()))


def crossing_step(steps_vals, threshold=0.95, ascending=True):
    """First step at which the value crosses threshold (and stays above)."""
    steps = list(steps_vals.keys())
    vals = [steps_vals[s] for s in steps]
    n = len(steps)
    for i in range(n):
        if ascending and vals[i] >= threshold and all(
            v >= threshold * 0.85 for v in vals[i:]
        ):
            return steps[i]
        if not ascending and vals[i] <= threshold and all(
            v <= threshold * 1.15 for v in vals[i:]
        ):
            return steps[i]
    return None


def llc_jumps(llc_series, k_sigma=3.0):
    """Identify centered 2nd-difference outliers in log(step) space.

    Sigma is the propagated chain-noise floor (sqrt(6) * mean per-step std).
    """
    steps = np.array(list(llc_series.keys()))
    means = np.array([llc_series[int(s)]["llc_mean"] for s in steps])
    stds = np.array([llc_series[int(s)]["llc_std"] for s in steps])
    if len(means) < 3:
        return []
    # Compute 2nd diff in log-step space (uniform spacing in log helps)
    log_steps = np.log(np.maximum(steps, 1))
    # 2nd derivative finite-difference w.r.t. log-step
    d2 = np.zeros_like(means)
    d2[:] = np.nan
    for i in range(1, len(means) - 1):
        h1 = log_steps[i] - log_steps[i - 1]
        h2 = log_steps[i + 1] - log_steps[i]
        # f''(x) ~ 2 * (h1*f[i+1] - (h1+h2)*f[i] + h2*f[i-1]) / (h1*h2*(h1+h2))
        d2[i] = 2 * (
            h1 * means[i + 1] - (h1 + h2) * means[i] + h2 * means[i - 1]
        ) / (h1 * h2 * (h1 + h2))
    # Noise floor from chain stds (each measurement ~sigma; 2nd diff ~sqrt(6)*sigma)
    sigma_floor = np.sqrt(6.0) * float(np.mean(stds))
    jumps = []
    for i in range(1, len(means) - 1):
        if abs(d2[i]) >= k_sigma * sigma_floor:
            jumps.append(
                (
                    int(steps[i]),
                    float(d2[i]),
                    float(d2[i] / sigma_floor),
                )
            )
    return jumps


def main():
    geom = parse_geom()
    global_llc, rllc = parse_llc()

    print("\n=== Geometry: best-layer order_score across training ===\n")
    print(f"{'step':>8}  " + "  ".join(f"{s:>22}" for s in geom))
    sizes = list(geom.keys())
    all_steps = sorted({s for size in sizes for s in geom[size]})
    for step in all_steps:
        row = [f"{step:>8}"]
        for size in sizes:
            d = geom[size].get(step)
            if d is None:
                row.append(f"{'':>22}")
            else:
                row.append(
                    f"L{d['best_layer']:>2} ord={d['order_score']:.2f} r/r={d['resid_r']:.2f}"
                )
        print("  ".join(row))

    print("\n=== Geometric formation point t* per size (order_score crosses 0.95) ===")
    formation = {}
    for size, series in geom.items():
        ord_series = {s: d["order_score"] for s, d in series.items()}
        t_star = crossing_step(ord_series, threshold=0.95, ascending=True)
        formation[size] = t_star
        print(f"  {size:>22}: t* = {t_star}")

    print("\n=== Pythia-160m: global LLC across training ===")
    print(f"{'step':>8}  {'llc_mean':>10}  {'llc_std':>9}  {'init_loss':>10}")
    for step, rec in global_llc.items():
        print(
            f"{step:>8}  {rec['llc_mean']:>10.3f}  {rec['llc_std']:>9.3f}  "
            f"{rec['init_loss']:>10.3f}"
        )

    print("\n=== Pythia-160m: rLLC at layers.7. across training ===")
    print(f"{'step':>8}  {'rllc_mean':>10}  {'rllc_std':>9}  {'init_loss':>10}  {'active_params':>14}")
    for step, rec in rllc.items():
        print(
            f"{step:>8}  {rec['llc_mean']:>10.3f}  {rec['llc_std']:>9.3f}  "
            f"{rec['init_loss']:>10.3f}  {rec['active_params']:>14}"
        )

    print("\n=== LLC stage candidates (centered 2nd-difference > 2 sigma) ===")
    for label, series in [("global", global_llc), ("rllc(layer7)", rllc)]:
        jumps = llc_jumps(series, k_sigma=2.0)
        print(f"  {label}: {len(jumps)} candidates")
        for step, val, z in jumps:
            print(f"    step={step:>7}  d2={val:>+10.2f}  ({z:+.2f} sigma)")

    print("\n=== Alignment test (Pythia-160m) ===")
    t_star = formation.get("pythia-160m")
    if t_star is None:
        print("  geometric formation t* not identified -- skipping alignment")
    else:
        global_jumps = llc_jumps(global_llc, k_sigma=2.0)
        rllc_jumps_ = llc_jumps(rllc, k_sigma=2.0)

        def closest(jumps):
            if not jumps:
                return None, None
            best = min(jumps, key=lambda j: abs(j[0] - t_star))
            return best, abs(best[0] - t_star)

        gj, gd = closest(global_jumps)
        rj, rd = closest(rllc_jumps_)
        print(f"  t* (geometric formation) = {t_star}")
        if gj:
            print(f"  closest global LLC jump  = step {gj[0]}  ({gj[2]:+.2f}sig)  |dt| = {gd}")
        if rj:
            print(f"  closest rLLC jump        = step {rj[0]}  ({rj[2]:+.2f}sig)  |dt| = {rd}")

    summary = {
        "formation_per_size": formation,
        "global_llc_jumps": llc_jumps(global_llc, k_sigma=2.0),
        "rllc_jumps": llc_jumps(rllc, k_sigma=2.0),
        "global_llc_series": global_llc,
        "rllc_series": rllc,
        "geom_summary": {
            size: {
                step: {
                    k: v
                    for k, v in d.items()
                    if k != "per_layer"
                }
                for step, d in series.items()
            }
            for size, series in geom.items()
        },
    }
    Path("data/summary.json").write_text(json.dumps(summary, indent=2))
    print("\nWrote data/summary.json")


if __name__ == "__main__":
    main()
