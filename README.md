# Developmental geometry of the day-of-week circle in Pythia

A small proof-of-concept at the intersection of **developmental interpretability**
(SLT / local learning coefficient) and **manifold geometry** (the days-of-week
circle). It watches a specific curved feature *form* across a model's training and
asks whether the loss landscape registers the same event the geometry does.

## Result

The days-of-week circle is one of the *earliest* features Pythia develops,
forming within the first **~256–1000 steps**. In Pythia-160M its formation
coincides — to within the checkpoint resolution — with an inflection in the
**refined LLC at the layer where the circle lives**, but not with any
discontinuity in the global LLC.

| | 70M | 160M | 410M | 1B |
|---|---|---|---|---|
| Formation step t\* (`order_score` crosses 0.95) | 512 | 256 | 256 | 1000 |

```
t* (geometric formation) = 256
closest global LLC jump   = step 128   (-2.17σ)   |Δt| = 128
closest rLLC(layer 7) jump = step 256  (-2.52σ)   |Δt| = 0   ← coincident
```

The rLLC inflection lands exactly on the geometric formation point; global LLC
shows no co-located event. The signal is modest (-2.5σ) — tightening it past the
checkpoint spacing would need denser checkpoints than Pythia publishes.

## How it works

- **Object**: 7 day centroids in the residual stream on Engels-style prompts
  (`"Let's do some day of the week math. {N} days from {Day} is"`), grouped by
  *target* day. PCA across the 7 centroids; fit a circle in the best 2-PC plane;
  report `order_score` (cyclic ordering = 1.0) and `resid/r`.
- **Developmental metric**: LLC via SGLD ([devinterp v2](https://github.com/timaeus-research/devinterp)),
  and refined LLC ([rLLC](https://arxiv.org/abs/2410.02984)) sampling only one
  layer's params. Fixed `lr=1e-4`, `localization=100`, 4 chains, 100 draws.
- **Models**: Pythia 70M/160M/410M/1B at 52 checkpoints; LLC time series on
  160M only.

Background: the day-circle is a causal 2D manifold used for weekday arithmetic
([Engels 2024](https://arxiv.org/abs/2405.14860), [Wurgaft 2026](https://arxiv.org/abs/2605.05115));
Pythia's 154 checkpoints ([suite](https://arxiv.org/abs/2304.01373)) let us watch
it form. LLC is an SLT quantity that detects when training enters a new region of
the loss landscape ([Hoogland 2024](https://arxiv.org/abs/2402.02364)); rLLC
([Wang 2024](https://arxiv.org/abs/2410.02984)) localizes that to one layer.

## rLLC's three phases (Pythia-160M)

Global LLC rises smoothly; rLLC at layer 7 shows structure it hides:

| Step | global LLC | rLLC(layer 7) | phase |
|---:|---:|---:|---|
| 1 | 5.16 | -0.14 | |
| **256** | **17.23** | **10.94** | rapid rise (→13 by 1k) |
| 5000 | 32.03 | 13.34 | plateau (~13–15 to 30k) |
| 80000 | 62.67 | 20.40 | late acceleration |
| 143000 | 156.94 | 49.11 | (→49) |

## Reproduce

**The analysis** runs locally from the committed `data/` — no GPU, no accounts:

```bash
pip install -r requirements.txt
python3 analyze.py          # reproduces the result above, writes data/summary.json
```

**The data** regenerates on Modal (needs a GPU-enabled account, ~60 min):

```bash
./run.sh                    # deploy → spawn → poll → analyze
```

LLC *absolute* values are SGLD seed/hardware-sensitive; the shape and the
alignment reproduce, the exact ±values drift. `order_score` is coarse (a 7-bit
quantity) — `resid/r` in `data/summary.json` tells the same story, finer-grained.

## Layout

| Path | Purpose |
|---|---|
| `circle_geometry.py` | Modal app: geometry + (r)LLC at a checkpoint |
| `orchestrate.py` | Spawn / poll the parallel Modal runs |
| `analyze.py` | Read result JSONs, run the alignment, write `data/summary.json` |
| `run.sh` | One-shot regenerate: deploy → spawn → poll → analyze |
| `data/` | Committed results (geometry, LLC time series, calibration, summary) |
| `requirements.txt` / `requirements-image.txt` | Local deps / remote GPU-image deps |
