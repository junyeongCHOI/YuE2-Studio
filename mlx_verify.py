"""Compare the MLX 8-bit AR path against the stock PyTorch backend.

Each backend runs in its own subprocess -- a BF16 MPS model, an FP32 CPU
reference and the MLX 8-bit model do not fit in memory at once -- and writes its
logits and greedy continuation to a scratch file. The final stage reads them
back and reports agreement and speed.

Both backends run behind sampling_guard, exactly as the server does, and the
guard's repair tally is reported per run: MPS does occasionally produce
non-finite logits mid-decode, and a comparison that hides that is not a fair one.

    .venv/bin/python mlx_verify.py                    # all three, 64 greedy tokens
    .venv/bin/python mlx_verify.py --skip-cpu         # no FP32 reference
    .venv/bin/python mlx_verify.py --tokens 256
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "YuE2-3B-mlx-8bit"


def build_prompts(take: Path | None, tokens: int):
    """An ABC-phase prefix and, when a saved take is available, a semantic one."""
    from yue2.protocol import SongRequest, token_prefixes
    from yue2.storage import resolve_model
    from yue2.tokenization_yue2 import YuE2TextTokenizer

    tokenizer = YuE2TextTokenizer(Path(resolve_model("m-a-p/YuE2-3B", local_files_only=True)) / "qwen.tiktoken")
    lyrics = (ROOT / "examples" / "lyrics.txt").read_text()
    request = SongRequest(style="Funk / nu-disco, warm female vocal, 110 BPM", lyrics=lyrics, cot="full", seed=123)
    prompts = [{
        "name": "abc", "phase": "abc",
        "prefix": token_prefixes(request, tokenizer),
        "sampling": {"temperature": 0.0, "top_p": 0.9, "top_k": 30, "repetition_penalty": 1.005,
                     "penalty_window": 100, "min_tokens": 0, "max_tokens": tokens},
    }]
    if take is not None:
        prefix = np.load(take / "prefix.npy").tolist()
        prompts.append({
            "name": "semantic", "phase": "semantic", "prefix": [int(t) for t in prefix],
            "sampling": {"temperature": 0.0, "top_p": 0.95, "top_k": 100, "repetition_penalty": 1.2,
                         "penalty_window": 50, "min_tokens": 0, "max_tokens": tokens},
        })
    return prompts


def greedy(generate, model, prompt, sampling):
    """Run one prompt, surviving the non-finite logits MPS sometimes emits."""
    import sampling_guard
    sampling_guard.take_repairs()
    failure = None
    try:
        tokens, timing, _ = generate(model, prompt["prefix"], sampling, 0, prompt["phase"])
    except RuntimeError as exc:
        tokens, timing, failure = [], {}, f"{type(exc).__name__}: {exc}"
    return {"name": prompt["name"], "tokens": tokens, "timing": timing,
            "repairs": sampling_guard.take_repairs(), "failure": failure}


# ── per-backend stages ────────────────────────────────────────────────────────


def run_mlx(prompts):
    import mlx.core as mx
    import mlx_ar
    import mlx_yue2
    from yue2.protocol import Sampling

    start = time.perf_counter()
    model, _ = mlx_yue2.load(MODEL_DIR)
    load_seconds = time.perf_counter() - start
    results = []
    for prompt in prompts:
        logits, caches = mlx_ar._prefill(model, prompt["prefix"])
        row = np.array(logits[0, -1].astype(mx.float32), copy=True)
        del caches
        run = greedy(mlx_ar.generate_tokens, model, prompt, Sampling(**prompt["sampling"]))
        results.append({"logits": row, **run})
    return results, load_seconds


def run_torch(prompts, device):
    import torch
    from yue2.modeling_yue2 import StaticKVCache, YuE2ForCausalLM
    from yue2.protocol import Sampling
    from yue2.sampling import generate_tokens
    from yue2.storage import resolve_model

    dtype = torch.bfloat16 if device == "mps" else torch.float32
    start = time.perf_counter()
    model = YuE2ForCausalLM.from_pretrained(resolve_model("m-a-p/YuE2-3B", local_files_only=True),
                                            local_files_only=True, dtype=dtype,
                                            low_cpu_mem_usage=True).eval().to(device)
    load_seconds = time.perf_counter() - start
    results = []
    with torch.inference_mode():
        for prompt in prompts:
            cache = StaticKVCache(num_layers=model.config.num_hidden_layers, batch_size=1,
                                  num_kv_heads=model.config.num_key_value_heads,
                                  max_seq_len=len(prompt["prefix"]) + 8,
                                  head_dim=model.config.head_dim, dtype=dtype,
                                  device=torch.device(device))
            output = model(torch.tensor([prompt["prefix"]], device=device), past_key_values=cache,
                           use_cache=True, logits_to_keep=1)
            row = output.logits[0, -1].float().cpu().numpy().copy()
            del cache, output
            run = greedy(lambda *a: generate_tokens(*a, use_cuda_graph=False), model, prompt,
                         Sampling(**prompt["sampling"]))
            results.append({"logits": row, **run})
    return results, load_seconds


def stage(name, prompts, out_path):
    import sampling_guard
    sampling_guard.install()
    runner = {"mlx": lambda: run_mlx(prompts),
              "mps": lambda: run_torch(prompts, "mps"),
              "cpu": lambda: run_torch(prompts, "cpu")}[name]
    results, load_seconds = runner()
    np.savez(out_path,
             meta=json.dumps({"backend": name, "load_seconds": load_seconds,
                              "runs": [{k: v for k, v in r.items() if k != "logits"} for r in results]}),
             **{r["name"]: r["logits"] for r in results})


# ── comparison ────────────────────────────────────────────────────────────────


def summarize(reference, other, k=10):
    """Logit-level agreement between two backends at one position."""
    reference, other = reference.astype(np.float64), other.astype(np.float64)
    order_ref, order_other = np.argsort(-reference), np.argsort(-other)
    p = np.exp(reference - reference.max())
    q = np.exp(other - other.max())
    p, q = p / p.sum(), q / q.sum()
    return {
        "top1_same": bool(order_ref[0] == order_other[0]),
        "topk_overlap": f"{len(set(order_ref[:k]) & set(order_other[:k]))}/{k}",
        "max_abs_diff": float(np.abs(reference - other).max()),
        # Elementwise rather than a dot product: numpy's matmul raises spurious
        # floating-point warnings on Accelerate for arrays this size.
        "cosine": float(np.sum(reference * other)
                        / np.sqrt(np.sum(reference * reference) * np.sum(other * other))),
        "kl_nats": float((p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum()),
    }


def common_prefix(a, b):
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def describe_run(run):
    repairs = run.get("repairs") or {}
    note = ""
    if repairs.get("steps"):
        note = f"  [guard repaired {repairs['tokens']} logits over {repairs['steps']} steps]"
    if run.get("failure"):
        note += f"  [ABORTED: {run['failure'].splitlines()[0]}]"
    return note


def compare(loaded, names):
    reference = "cpu" if "cpu" in loaded else "mps"
    print(f"\nreference backend: {reference} "
          f"({'float32 CPU' if reference == 'cpu' else 'bfloat16 MPS'})\n")
    for run_name in names:
        print("=" * 78)
        print(f"prompt: {run_name}")
        print("=" * 78)
        base_logits = loaded[reference]["logits"][run_name]
        base_run = loaded[reference]["meta"][run_name]
        print("  logits at the first generated position, vs the reference:")
        for backend, data in loaded.items():
            if backend == reference:
                continue
            stats = summarize(base_logits, data["logits"][run_name])
            print(f"    {backend:4s} top1 {'same' if stats['top1_same'] else 'DIFFERENT'}, "
                  f"top10 {stats['topk_overlap']}, cos {stats['cosine']:.6f}, "
                  f"max|Δ| {stats['max_abs_diff']:.4f}, KL {stats['kl_nats']:.2e}")
        print(f"  greedy continuation ({len(base_run['tokens'])} reference tokens):")
        for backend, data in loaded.items():
            run = data["meta"][run_name]
            if backend == reference:
                print(f"    {backend:4s} reference{describe_run(run)}")
                continue
            match = common_prefix(base_run["tokens"], run["tokens"])
            detail = (f"identical for {match}/{len(base_run['tokens'])} tokens"
                      if not run["failure"] else "no tokens")
            if base_run["tokens"] and match < len(base_run["tokens"]) and not run["failure"]:
                detail += f" (first difference at step {match})"
            print(f"    {backend:4s} {detail}{describe_run(run)}")
        print("  speed:")
        for backend, data in loaded.items():
            timing = data["meta"][run_name]["timing"]
            if not timing:
                print(f"    {backend:4s} no timing (run aborted)")
                continue
            print(f"    {backend:4s} prefill {timing['prefill_seconds']:6.2f}s "
                  f"({timing['prefix_tokens']} tokens)   decode {timing['output_tps']:6.2f} tok/s   "
                  f"load {data['load_seconds']:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--take", help="library directory with prefix.npy for a realistic semantic prompt")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--skip-cpu", action="store_true", help="skip the FP32 CPU reference")
    parser.add_argument("--stage", choices=("mlx", "mps", "cpu"), help=argparse.SUPPRESS)
    parser.add_argument("--work", help=argparse.SUPPRESS)
    args = parser.parse_args()

    take = Path(args.take) if args.take else next(iter(sorted(
        p.parent for p in ROOT.glob("outputs/library/*/prefix.npy"))), None)
    prompts = build_prompts(take, args.tokens)

    if args.stage:
        stage(args.stage, prompts, Path(args.work) / f"{args.stage}.npz")
        return 0

    print(f"model:   {MODEL_DIR}")
    print(f"prompts: " + ", ".join(f"{p['name']} ({len(p['prefix'])} tokens)" for p in prompts))
    print(f"greedy:  {args.tokens} tokens per backend")
    backends = ["mlx", "mps"] + ([] if args.skip_cpu else ["cpu"])
    work = Path(tempfile.mkdtemp(prefix="yue2-mlx-verify-"))
    loaded = {}
    for backend in backends:
        print(f"\n==> {backend}", flush=True)
        started = time.perf_counter()
        subprocess.run([sys.executable, __file__, "--stage", backend, "--work", str(work),
                        "--tokens", str(args.tokens)] + (["--take", str(take)] if take else []),
                       check=True, cwd=ROOT)
        print(f"    {time.perf_counter() - started:.1f}s")
        data = np.load(work / f"{backend}.npz")
        meta = json.loads(str(data["meta"]))
        loaded[backend] = {"logits": {p["name"]: data[p["name"]] for p in prompts},
                           "meta": {run["name"]: run for run in meta["runs"]},
                           "load_seconds": meta["load_seconds"]}
    compare(loaded, [p["name"] for p in prompts])
    print(f"\nscratch: {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
