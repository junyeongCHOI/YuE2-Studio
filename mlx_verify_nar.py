"""Compare MLX acoustic synthesis against the stock PyTorch NAR path.

Unlike token generation this is fully deterministic: the same semantic tokens
and the same seeded CPU noise draw must produce the same latents, so any
difference is numerical. Both backends run in their own subprocess, then the
latents are decoded with the same PyTorch VAE and the waveforms compared --
the measure the README already uses for ODE step counts.

    .venv/bin/python mlx_verify_nar.py                 # 32 steps, first saved take
    .venv/bin/python mlx_verify_nar.py --steps 8
    .venv/bin/python mlx_verify_nar.py --take outputs/library/<id>
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


def load_take(take: Path):
    request = json.loads((take / "request.json").read_text())
    return {"prefix": [int(t) for t in np.load(take / "prefix.npy")],
            "codec": [int(t) for t in np.load(take / "semantic.npy")],
            "seed": int(request["seed"])}


def run_torch(inputs, steps, device="mps"):
    import torch
    from yue2.nar import synthesize
    from yue2.storage import resolve_model
    from yue2.modeling_yue2 import YuE2ForCausalLM

    dtype = torch.bfloat16 if device == "mps" else torch.float32
    start = time.perf_counter()
    model = YuE2ForCausalLM.from_pretrained(resolve_model("m-a-p/YuE2-3B", local_files_only=True),
                                            local_files_only=True, dtype=dtype,
                                            low_cpu_mem_usage=True).eval().to(device)
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    latents = synthesize(model, inputs["prefix"], inputs["codec"], inputs["seed"], steps=steps)
    return latents.float().cpu().numpy(), time.perf_counter() - start, load_seconds


def run_mlx(inputs, steps, model_dir):
    import mlx_nar
    import mlx_yue2

    start = time.perf_counter()
    model, _ = mlx_yue2.load(model_dir)
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    latents = mlx_nar.synthesize(model, inputs["prefix"], inputs["codec"], inputs["seed"], steps=steps)
    return latents.numpy(), time.perf_counter() - start, load_seconds


def decode(latents):
    """Same VAE, same tiling, for both sets of latents."""
    import torch
    from yue2.modeling_vae import YuE2VAE
    from yue2.storage import resolve_model

    vae = YuE2VAE.from_pretrained(resolve_model("m-a-p/YuE2-Vae", local_files_only=True),
                                  decoder_only=True, device="cpu", local_files_only=True).to("mps")
    out = []
    with torch.inference_mode():
        for array in latents:
            z = torch.as_tensor(array, dtype=torch.float32).T.unsqueeze(0)
            audio = vae.decode_tiled(z, core_frames=1024, halo_frames=16, output_device="cpu")
            out.append(audio[0].float().clamp(-1, 1).T.contiguous().numpy())
    return out


def correlation(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    # Elementwise: numpy's matmul raises spurious FP warnings on Accelerate.
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


def envelope(audio, sr=48000, window=480):
    mono = audio.mean(1)
    usable = len(mono) // window * window
    return np.sqrt((mono[:usable].reshape(-1, window) ** 2).mean(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--take", help="library directory with prefix.npy + semantic.npy")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--model", default=str(MODEL_DIR), help="the MLX model under test")
    parser.add_argument("--against", default=None, metavar="DIR",
                        help="compare against another MLX model instead of PyTorch, "
                             "e.g. a --nar-bf16 conversion to isolate quantization error")
    parser.add_argument("--with-cpu", action="store_true",
                        help="also run an FP32 CPU reference (slow) and score both against it")
    parser.add_argument("--stage", choices=("torch", "mlx", "cpu", "mlx_ref"), help=argparse.SUPPRESS)
    parser.add_argument("--work", help=argparse.SUPPRESS)
    args = parser.parse_args()

    take = Path(args.take) if args.take else next(iter(sorted(
        p.parent for p in ROOT.glob("outputs/library/*/semantic.npy"))), None)
    if take is None:
        raise SystemExit("No saved take found; pass --take")
    inputs = load_take(take)

    if args.stage:
        runner = {"torch": lambda: run_torch(inputs, args.steps, "mps"),
                  "cpu": lambda: run_torch(inputs, args.steps, "cpu"),
                  "mlx": lambda: run_mlx(inputs, args.steps, args.model),
                  "mlx_ref": lambda: run_mlx(inputs, args.steps, args.against)}[args.stage]
        latents, seconds, load_seconds = runner()
        np.savez(Path(args.work) / f"{args.stage}.npz", latents=latents,
                 meta=json.dumps({"seconds": seconds, "load_seconds": load_seconds}))
        return 0

    print(f"model:  {args.model}")
    print(f"take:   {take}  ({len(inputs['codec'])} codec frames, prefix {len(inputs['prefix'])})")
    print(f"steps:  {args.steps} midpoint ({2 * args.steps} velocity evaluations)")
    work = Path(tempfile.mkdtemp(prefix="yue2-mlx-nar-"))
    loaded = {}
    backends = ["mlx", "mlx_ref" if args.against else "torch"] + (["cpu"] if args.with_cpu else [])
    for backend in backends:
        print(f"\n==> {backend}" + (f"  {args.against}" if backend == "mlx_ref" else ""), flush=True)
        subprocess.run([sys.executable, __file__, "--stage", backend, "--work", str(work),
                        "--steps", str(args.steps), "--take", str(take), "--model", args.model]
                       + (["--against", args.against] if args.against else []), check=True, cwd=ROOT)
        data = np.load(work / f"{backend}.npz")
        loaded[backend] = {"latents": data["latents"], **json.loads(str(data["meta"]))}

    reference = "cpu" if "cpu" in loaded else ("mlx_ref" if "mlx_ref" in loaded else "torch")
    others = [b for b in loaded if b != reference]
    print("\n" + "=" * 70)
    labels = {"cpu": "float32 CPU", "torch": "bfloat16 MPS", "mlx_ref": f"MLX {args.against}"}
    print(f"latents vs {reference} ({labels[reference]})")
    print("=" * 70)
    base = loaded[reference]["latents"]
    print(f"  shape {base.shape}, |x| mean {np.abs(base).mean():.4f}")
    for backend in others:
        other = loaded[backend]["latents"]
        print(f"  {backend:5s} correlation {correlation(base, other):.6f}   "
              f"max abs diff {np.abs(base - other).max():.4f}   "
              f"relative RMS {np.sqrt(((base - other) ** 2).mean()) / np.sqrt((base ** 2).mean()):.3%}")

    print("\n" + "=" * 70)
    print("audio (same PyTorch VAE on every set of latents)")
    print("=" * 70)
    names = [reference] + others
    decoded = dict(zip(names, decode([loaded[n]["latents"] for n in names])))
    for backend in others:
        print(f"  {backend:5s} waveform correlation {correlation(decoded[reference], decoded[backend]):.4f}   "
              f"envelope {correlation(envelope(decoded[reference]), envelope(decoded[backend])):.4f}")
    for name, audio in decoded.items():
        np.save(work / f"audio_{name}.npy", audio)
        print(f"  {name:5s} peak {np.abs(audio).max():.3f}  rms {np.sqrt((audio ** 2).mean()):.4f}")

    print("\n" + "=" * 70)
    print("speed")
    print("=" * 70)
    for backend, data in loaded.items():
        print(f"  {backend:5s} synthesis {data['seconds']:6.1f}s   model load {data['load_seconds']:.1f}s")
    print(f"\nscratch: {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
