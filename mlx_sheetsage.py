"""SheetSage2's audio encoder (MERT-v2 + layer mix) on MLX.

Transcription spends its memory and much of its time in the encoder: a
24-block, 1024-wide Conformer over every 300 s window at 25 frames/s. This file
runs that half on MLX; the 57M-parameter decoder and its grammar-constrained
search stay in PyTorch on the CPU (sheetsage_runner.py wires the two).

Runs in .venv-sheetsage, which has both torch (to read and LoRA-merge the
checkpoint once) and mlx:

    .venv-sheetsage/bin/python mlx_sheetsage.py convert          # -> models/SheetSage2-encoder-mlx
    .venv-sheetsage/bin/python mlx_sheetsage.py verify <audio>   # against the PyTorch FP32 encoder

Precision. Weights are stored in FP16 and every activation is computed in FP32.
The decoder reads this output token by token for thousands of steps, and on a
253 s song a fully half-precision encoder (BF16 or FP16) sent it down another
path within the first 12% of tokens -- a different score. FP16 weights with FP32
arithmetic gave the FP32 reference's ABC exactly (3523 of 3535 tokens equal;
the rest were beat times 10-30 ms apart), at FP16's 1.2 GB.

The log-mel front end stays in PyTorch: it is a fixed STFT and filterbank, cheap
on the CPU, and keeping it there keeps its numerics exactly SheetSage2's.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parent
SHEETSAGE = ROOT / "sheetsage2"
DEFAULT_DIR = ROOT / "models" / "SheetSage2-encoder-mlx"
# Not model.safetensors: mlx_backend.discover() lists directories holding one as YuE2 models.
WEIGHTS = "encoder.safetensors"
CONFIG = "config.json"
COMPUTE = mx.float32
# Query frames per attention slice. The window is 7500 frames and an unsliced
# attention matrix is 16 x 7500 x 7500 -- 3.35 GiB in FP32. Rows of unmasked,
# non-causal attention are independent, so slicing the queries changes nothing
# but the peak (checked bit-identical).
ATTENTION_SLICE = 1024


# ── conversion ────────────────────────────────────────────────────────────────


def load_torch_model(source=SHEETSAGE):
    """SheetSage2 in FP32 on the CPU, LoRA merged -- the reference weights."""
    import torch
    from transformers import AutoModel
    return AutoModel.from_pretrained(str(source), trust_remote_code=True,
                                     torch_dtype=torch.float32).eval()


def export_tensors(model):
    """Encoder tensors under their PyTorch names, conv kernels made channels-last."""
    tensors = {}
    for name, value in model.state_dict().items():
        if name.startswith("encoder.feature_extractor."):
            continue                               # the front end stays in PyTorch
        if not name.startswith(("encoder.", "layer_weight", "encoder_projection.")):
            continue
        array = value.detach().float().numpy()
        # Conv1d kernels (out, in/groups, K) -> MLX's (out, K, in/groups). The global
        # response norm's (1, 1, C) scale is also 3-D but is not a kernel.
        if array.ndim == 3 and name.endswith(".weight") and not name.endswith("pointwise_block.3.weight"):
            array = array.transpose(0, 2, 1)
        tensors[name.removeprefix("encoder.")] = array
    return tensors


def convert(source=SHEETSAGE, destination=DEFAULT_DIR, dtype="float16"):
    start = time.perf_counter()
    model = load_torch_model(source)
    config = {
        "backbone": {key: model.encoder.config.to_dict()[key] for key in (
            "hidden_size", "num_attention_heads", "num_hidden_layers", "rotary_embedding_base",
            "layer_norm_eps", "subsampling_layer_norm_eps", "subsampling_channels", "num_mel_bins")},
        "mel_mean": model.encoder.feature_extractor.mel_mean.tolist(),
        "mel_std": model.encoder.feature_extractor.mel_std.tolist(),
        "source": {"path": str(source), "base_model_sha256": model.config.base_model_sha256},
        "dtype": dtype,
    }
    tensors = export_tensors(model)
    del model
    gc.collect()

    arrays = {}
    for name in list(tensors):
        array = mx.array(tensors.pop(name))       # each PyTorch buffer is released as it goes
        arrays[name] = array if name == "layer_weight" else array.astype(getattr(mx, dtype))
        mx.eval(arrays[name])

    destination = Path(destination)
    staging = destination.parent / f".{destination.name}.partial"
    staging.mkdir(parents=True, exist_ok=True)
    for child in staging.iterdir():
        child.unlink()
    mx.save_safetensors(str(staging / WEIGHTS), arrays, metadata={"format": "mlx"})
    (staging / CONFIG).write_text(json.dumps(config, indent=2) + "\n")
    if destination.exists():
        for child in destination.iterdir():
            child.unlink()
        destination.rmdir()
    staging.rename(destination)
    size = (destination / WEIGHTS).stat().st_size
    print(f"wrote {destination} ({size / 2**30:.2f} GiB, {time.perf_counter() - start:.1f}s)")


# ── model ─────────────────────────────────────────────────────────────────────


def available(directory=DEFAULT_DIR):
    directory = Path(directory)
    return (directory / WEIGHTS).is_file() and (directory / CONFIG).is_file()


def load(directory=DEFAULT_DIR):
    directory = Path(directory)
    config = json.loads((directory / CONFIG).read_text())
    weights = mx.load(str(directory / WEIGHTS))
    mx.eval(list(weights.values()))
    return weights, config


def param(weights, name):
    """A stored tensor in the compute dtype; each op casts only the weights it uses."""
    value = weights.get(name)
    return None if value is None else value.astype(COMPUTE)


def linear(weights, name, x):
    y = x @ param(weights, f"{name}.weight").T
    bias = param(weights, f"{name}.bias")
    return y if bias is None else y + bias


def layer_norm(weights, name, x, eps):
    return mx.fast.layer_norm(x, param(weights, f"{name}.weight"), param(weights, f"{name}.bias"), eps)


def conv1d(weights, name, x, *, stride=1, padding=0, groups=1):
    y = mx.conv1d(x, param(weights, f"{name}.weight"), stride=stride, padding=padding, groups=groups)
    bias = param(weights, f"{name}.bias")
    return y if bias is None else y + bias


def convnext_layer(weights, name, x, eps):
    y = conv1d(weights, f"{name}.depthwise_block.1", x, padding=3, groups=x.shape[-1])
    y = layer_norm(weights, f"{name}.pointwise_block.0", y, eps)
    y = nn.gelu(linear(weights, f"{name}.pointwise_block.1", y))
    # Global response norm: the L2 norm runs over time, across the whole window.
    magnitude = mx.sqrt(mx.sum(y * y, axis=1, keepdims=True))
    normalized = magnitude / (mx.mean(magnitude, axis=-1, keepdims=True) + 1e-6)
    y = param(weights, f"{name}.pointwise_block.3.weight") * (y * normalized) \
        + param(weights, f"{name}.pointwise_block.3.bias") + y
    return x + linear(weights, f"{name}.pointwise_block.4", y)


def subsample(weights, config, x):
    eps = config["backbone"]["subsampling_layer_norm_eps"]
    channels = [config["backbone"]["num_mel_bins"]] + config["backbone"]["subsampling_channels"]
    for index, stride in enumerate((1, 2, 2)):
        name = f"subsampling_module.{index}"
        if channels[index] != channels[index + 1] or stride > 1:
            x = layer_norm(weights, f"{name}.resampling_layer.0", x, eps)
            x = conv1d(weights, f"{name}.resampling_layer.2", x, stride=stride)
        depth = 0
        while f"{name}.convnext_layers.{depth}.depthwise_block.1.weight" in weights:
            x = convnext_layer(weights, f"{name}.convnext_layers.{depth}", x, eps)
            mx.eval(x)                             # one layer's intermediates at a time
            depth += 1
    return x


def rotary(config, length):
    width, heads = config["backbone"]["hidden_size"], config["backbone"]["num_attention_heads"]
    head_dim = width // heads
    inverse = 1.0 / (config["backbone"]["rotary_embedding_base"]
                     ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    angles = mx.arange(length, dtype=mx.float32)[:, None] * inverse[None]
    angles = mx.concatenate([angles, angles], axis=-1)
    return mx.cos(angles)[:, None, :], mx.sin(angles)[:, None, :]


def rotate_half(x):
    first, second = mx.split(x, 2, axis=-1)
    return mx.concatenate([-second, first], axis=-1)


def attention(weights, name, x, heads, cos, sin):
    batch, length, width = x.shape
    shape = (batch, length, heads, width // heads)
    q, k, v = (linear(weights, f"{name}.{projection}", x).reshape(shape)
               for projection in ("query_proj", "key_proj", "value_proj"))
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
    slices = []
    for start in range(0, length, ATTENTION_SLICE):
        slices.append(mx.fast.scaled_dot_product_attention(
            q[:, :, start:start + ATTENTION_SLICE], k, v, scale=1 / math.sqrt(width // heads)))
        mx.eval(slices[-1])
    attended = mx.concatenate(slices, axis=2).transpose(0, 2, 1, 3).reshape(batch, length, width)
    return linear(weights, f"{name}.out_proj", attended)


def conformer_block(weights, config, name, x, cos, sin):
    eps = config["backbone"]["layer_norm_eps"]
    width = x.shape[-1]

    def ffn(prefix, y):
        return linear(weights, f"{prefix}.w_2", nn.gelu(linear(weights, f"{prefix}.w_1", y)))

    x = x + 0.5 * ffn(f"{name}.ffn1", layer_norm(weights, f"{name}.ffn1_layer_norm", x, eps))
    x = attention(weights, f"{name}.attn", layer_norm(weights, f"{name}.attn_layer_norm", x, eps),
                  config["backbone"]["num_attention_heads"], cos, sin) + x

    y = layer_norm(weights, f"{name}.conv_module.layer_norm", x, eps)
    gate, value = mx.split(conv1d(weights, f"{name}.conv_module.conv_block.1", y), 2, axis=-1)
    y = gate * mx.sigmoid(value)
    kernel = weights[f"{name}.conv_module.conv_block.3.weight"].shape[1]
    y = conv1d(weights, f"{name}.conv_module.conv_block.3", y, padding=(kernel - 1) // 2, groups=width)
    y = nn.gelu(layer_norm(weights, f"{name}.conv_module.conv_block.4.1", y, eps))
    x = conv1d(weights, f"{name}.conv_module.conv_block.6", y) + x

    x = x + 0.5 * ffn(f"{name}.ffn2", layer_norm(weights, f"{name}.ffn2_layer_norm", x, eps))
    return layer_norm(weights, f"{name}.final_layer_norm", x, eps)


def encode(weights, config, mel):
    """Normalised log-mel frames [batch, frames, bins] -> decoder memory [batch, frames / 4, 512].

    Mirrors SheetSage2Model.get_audio_features: every block's output is mixed
    by softmax(layer_weight) and projected. Each block is evaluated before the
    next is built, so the peak is one block's activations.
    """
    x = subsample(weights, config, mx.array(mel).astype(COMPUTE))
    mix = mx.softmax(param(weights, "layer_weight"), axis=0)
    mixed = x * mix[0]
    cos, sin = rotary(config, x.shape[1])
    for index in range(config["backbone"]["num_hidden_layers"]):
        x = conformer_block(weights, config, f"layers.{index}", x, cos, sin)
        mixed = mixed + x * mix[index + 1]
        mx.eval(x, mixed)
    return np.array(linear(weights, "encoder_projection", mixed))


# ── verification ──────────────────────────────────────────────────────────────


def verify(audio, directory=DEFAULT_DIR, max_seconds=30.0):
    """Compare decoder memory from MLX with the PyTorch FP32 encoder on one window."""
    import importlib
    import torch

    model = load_torch_model()
    package = type(model).__module__.rpartition(".")[0]
    audio_module = importlib.import_module(f"{package}.audio_sheetsage2")
    waveform = audio_module.load_audio(audio, max_seconds=max_seconds)
    segment = audio_module.slice_audio(waveform, 0, float(model.hparams.input_audio_length))[None]
    with torch.inference_mode():
        prepared, _ = model._prepare_audio(segment)
        mel = model.encoder.feature_extractor(prepared).numpy()
        start = time.perf_counter()
        reference = model.encode(segment).numpy()
        torch_seconds = time.perf_counter() - start
    del model
    gc.collect()

    mx.set_cache_limit(0)
    weights, config = load(directory)
    start = time.perf_counter()
    candidate = encode(weights, config, mel)
    mlx_seconds = time.perf_counter() - start
    a, b = reference.reshape(-1).astype(np.float64), candidate.reshape(-1).astype(np.float64)
    print(f"torch FP32 CPU  {torch_seconds:6.1f}s")
    print(f"MLX             {mlx_seconds:6.1f}s  peak {mx.get_peak_memory() / 2**30:.2f} GiB")
    print(f"cosine          {a @ b / np.linalg.norm(a) / np.linalg.norm(b):.8f}")
    print(f"relative RMS    {np.linalg.norm(a - b) / np.linalg.norm(a):.2e}")
    print(f"max abs error   {np.abs(a - b).max():.2e} (reference abs max {np.abs(a).max():.3f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("convert", help="export the LoRA-merged encoder to MLX")
    make.add_argument("--out", default=str(DEFAULT_DIR))
    make.add_argument("--dtype", default="float16", choices=("float16", "float32"),
                      help="storage precision (arithmetic is FP32 either way)")
    check = commands.add_parser("verify", help="compare against the PyTorch FP32 encoder")
    check.add_argument("audio")
    check.add_argument("--model", default=str(DEFAULT_DIR))
    check.add_argument("--max-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.command == "convert":
        convert(destination=args.out, dtype=args.dtype)
    else:
        verify(args.audio, args.model, args.max_seconds)


if __name__ == "__main__":
    main()
