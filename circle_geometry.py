"""
Project 1: developmental geometry of the days-of-week circle across Pythia training.

Three function types, all parallelizable:
  - geometry_at_checkpoint: collect day-prompt residual streams, compute circle metrics
  - prepare_eval_tokens:    one-time tokenize a fixed Pile slice for LLC
  - llc_at_checkpoint:      global LLC or rLLC via devinterp v2 (SGLD)
"""

import modal

app = modal.App("project1-dow-development")

hf_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("p1-data", create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "numpy==1.26.4",
        "scikit-learn==1.5.2",
        "scipy==1.14.1",
        "datasets==3.1.0",
        "zstandard",
    )
)

llc_image = base_image.pip_install("devinterp==2.0.1", "xarray>=2024.1.0", "zarr>=3.0.0")


# ---------------- checkpoint schedule (52 steps subsampled from 154) -------
LOG_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
EARLY_STEPS = list(range(1000, 30001, 1000))                  # 30 ckpts
LATE_STEPS = list(range(40000, 140001, 10000)) + [143000]      # 12 ckpts
ALL_STEPS = LOG_STEPS + EARLY_STEPS + LATE_STEPS               # 52 total


# ---------------- prompt set ----------------
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
OFFSET_WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six"]


def build_prompts():
    prompts, targets = [], []
    for i, start in enumerate(DAYS):
        for k, off_word in enumerate(OFFSET_WORDS):
            prompts.append(
                f"Let's do some day of the week math. {off_word} days from {start} is"
            )
            targets.append((i + k) % 7)
    return prompts, targets


def fit_circle_2d(xy):
    import numpy as np
    x, y = xy[:, 0], xy[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = float(sol[0]), float(sol[1])
    r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0)))
    if r <= 1e-9:
        return cx, cy, r, float("inf")
    residual = float(
        np.sqrt(np.mean((np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r) ** 2))
    )
    return cx, cy, r, residual


def cyclic_order_score(angles):
    import numpy as np
    order = np.argsort(angles)
    incs = np.diff(order) % 7
    if len(incs) == 0:
        return 0.0
    counts = np.bincount(incs, minlength=7)
    return float(counts.max() / len(incs))


# ============================================================
# Function 1: geometry at a single checkpoint
# ============================================================
@app.function(
    image=base_image,
    gpu="A10G",
    timeout=1500,
    volumes={"/cache": hf_volume},
    max_containers=10,
)
def geometry_at_checkpoint(model_name: str, step: int):
    import os
    os.environ["HF_HOME"] = "/cache/huggingface"

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.decomposition import PCA

    rev = f"step{step}"
    print(f"Loading {model_name} @ {rev}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=rev, torch_dtype=torch.float32
    ).cuda()
    model.eval()

    prompts, targets = build_prompts()
    targets_arr = np.array(targets)
    all_hidden = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.cuda()
            out = model(ids, output_hidden_states=True, use_cache=False)
            hs = torch.stack([h[0, -1].cpu() for h in out.hidden_states])
            all_hidden.append(hs.float().numpy())
    all_hidden = np.stack(all_hidden)
    n_layers = all_hidden.shape[1]
    d_model = all_hidden.shape[2]

    per_layer = []
    for layer in range(n_layers):
        centroids = np.stack(
            [all_hidden[targets_arr == d, layer].mean(0) for d in range(7)]
        )
        n_comp = min(6, d_model)
        pca = PCA(n_components=n_comp)
        proj = pca.fit_transform(centroids)
        evr = pca.explained_variance_ratio_
        var = pca.explained_variance_
        if float(var.sum()) > 0:
            p_ratio = float((var.sum() ** 2) / float((var ** 2).sum()))
        else:
            p_ratio = 0.0

        candidate_pairs = [(0, 1), (1, 2)] if n_comp >= 3 else [(0, 1)]
        best = None
        for a, b in candidate_pairs:
            xy = proj[:, [a, b]]
            cx, cy, r, resid = fit_circle_2d(xy)
            if r <= 1e-9:
                continue
            angles = np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx)
            order = cyclic_order_score(angles)
            ratio = resid / r
            score = order - ratio
            cand = {
                "pc_pair": [int(a), int(b)],
                "circle_residual_ratio": float(ratio),
                "order_score": float(order),
                "radius": float(r),
                "score": float(score),
                "pc_explained_var": [float(evr[a]), float(evr[b])],
            }
            if best is None or cand["score"] > best["score"]:
                best = cand

        per_layer.append(
            {
                "layer": int(layer),
                "best_pair": best,
                "evr_top6": [float(v) for v in evr],
                "participation_ratio": p_ratio,
            }
        )

    scored = [
        (p["best_pair"]["score"], p["layer"]) for p in per_layer if p["best_pair"]
    ]
    scored.sort(reverse=True, key=lambda t: t[0])
    best_layer = scored[0][1] if scored else None
    return {
        "model": model_name,
        "step": int(step),
        "n_layers": int(n_layers),
        "d_model": int(d_model),
        "best_layer": best_layer,
        "best_layer_metrics": (
            per_layer[best_layer]["best_pair"] if best_layer is not None else None
        ),
        "per_layer": per_layer,
    }


# ============================================================
# One-time eval data prep
# ============================================================
@app.function(
    image=base_image,
    cpu=2,
    timeout=1800,
    volumes={"/cache": hf_volume, "/data": data_volume},
)
def prepare_eval_tokens(seq_len: int = 256, n_sequences: int = 256):
    """Tokenize a fixed Pile slice once, save to /data for reuse."""
    import os
    os.environ["HF_HOME"] = "/cache/huggingface"

    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer

    out_path = f"/data/eval_tokens_{seq_len}_{n_sequences}.npy"
    if os.path.exists(out_path):
        arr = np.load(out_path)
        print(f"Already prepared: {arr.shape}", flush=True)
        return list(arr.shape)

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    print("Loading NeelNanda/pile-10k ...", flush=True)
    raw = load_dataset("NeelNanda/pile-10k", split="train")

    tokens: list[int] = []
    target = n_sequences * seq_len + seq_len
    for ex in raw:
        ids = tok(ex["text"]).input_ids
        if len(ids) > 16:
            tokens.extend(ids)
        if len(tokens) >= target:
            break

    if len(tokens) < n_sequences * seq_len:
        raise RuntimeError(
            f"Only got {len(tokens)} tokens; need {n_sequences * seq_len}"
        )
    arr = np.array(tokens[: n_sequences * seq_len], dtype=np.int64).reshape(
        n_sequences, seq_len
    )
    np.save(out_path, arr)
    data_volume.commit()
    print(f"Saved {arr.shape}", flush=True)
    return list(arr.shape)


# ============================================================
# LLC measurement (devinterp v2)
# ============================================================
@app.function(
    image=llc_image,
    gpu="A10G",
    timeout=3600,
    volumes={"/cache": hf_volume, "/data": data_volume},
    max_containers=10,
)
def llc_at_checkpoint(
    model_name: str,
    step: int,
    lr: float = 1e-4,
    localization: float = 100.0,
    n_beta: float | None = None,
    num_chains: int = 4,
    num_draws: int = 100,
    num_burnin_steps: int = 0,
    batch_size: int = 4,
    seq_len: int = 256,
    n_sequences: int = 256,
    layer_substring: str | None = None,
    init_seed: int = 100,
):
    """
    layer_substring=None  -> global LLC (all params optimized).
    layer_substring="layers.N." -> rLLC (only params containing this substring).
    """
    import os
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from devinterp.slt.llc import llc as llc_fn
    from devinterp.utils import default_nbeta, tokenize_and_concatenate

    # Build the eval dataset using devinterp's helper to ensure correct format.
    raw = load_dataset("NeelNanda/pile-10k", split="train").select(range(min(800, n_sequences * 4)))
    tok = AutoTokenizer.from_pretrained(model_name)
    ds = tokenize_and_concatenate(
        raw, tok, column_name="text", add_bos_token=False, max_length=seq_len, num_proc=2
    )
    ds = ds.select(range(min(n_sequences, len(ds))))
    print(f"eval dataset: {len(ds)} sequences of length {seq_len}", flush=True)

    rev = f"step{step}"
    print(f"Loading {model_name} @ {rev}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=rev, torch_dtype=torch.float32
    ).cuda()
    model.train()

    if layer_substring is not None:
        param_masks = {
            name: None
            for name, _ in model.named_parameters()
            if layer_substring in name
        }
        active_count = sum(
            int(p.numel())
            for n, p in model.named_parameters()
            if layer_substring in n
        )
        if active_count == 0:
            return {
                "model": model_name,
                "step": int(step),
                "layer_substring": layer_substring,
                "error": f"no params match substring {layer_substring!r}",
            }
    else:
        param_masks = None
        active_count = sum(int(p.numel()) for p in model.parameters())

    if n_beta is None:
        n_beta = default_nbeta(batch_size)

    print(
        f"layer={layer_substring} active={active_count} lr={lr} "
        f"loc={localization} n_beta={n_beta} chains={num_chains} draws={num_draws}",
        flush=True,
    )

    result = llc_fn(
        model=model,
        dataset=ds,
        observables={"train": ds},
        lr=lr,
        n_beta=n_beta,
        localization=localization,
        param_masks=param_masks,
        num_chains=num_chains,
        num_draws=num_draws,
        num_burnin_steps=num_burnin_steps,
        batch_size=batch_size,
        num_init_loss_batches=8,
        init_seed=init_seed,
    )

    out = {
        "model": model_name,
        "step": int(step),
        "layer_substring": layer_substring,
        "lr": float(lr),
        "localization": float(localization),
        "n_beta": float(n_beta),
        "num_chains": int(num_chains),
        "num_draws": int(num_draws),
        "num_burnin_steps": int(num_burnin_steps),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "n_sequences": int(n_sequences),
        "active_params": int(active_count),
        "llc_mean": float(result["llc_mean"]),
        "llc_std": float(result["llc_std"]),
        "llc_per_chain": [float(x) for x in result["llc_per_chain"].values],
        "init_loss": float(result["init_loss"]),
    }
    print(out, flush=True)
    return out


@app.local_entrypoint()
def main_one_geom(model: str = "EleutherAI/pythia-160m", step: int = 143000):
    import json
    res = geometry_at_checkpoint.remote(model, step)
    print(json.dumps(res["best_layer_metrics"], indent=2))
    print("best layer:", res["best_layer"])


@app.local_entrypoint()
def main_one_llc(model: str = "EleutherAI/pythia-160m", step: int = 71500):
    import json
    res = llc_at_checkpoint.remote(model, step)
    print(json.dumps(res, indent=2))
