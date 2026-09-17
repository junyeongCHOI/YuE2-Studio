"""Token generation on the MLX model: the ABC plan and the semantic stream.

Only the forward pass changes. Sampling is deliberately not reimplemented --
logits come back to torch and go through ``yue2.sampling.distribution`` and the
same seeded CPU multinomial the stock backend uses, so masking, repetition
penalties, truncation and token selection stay upstream's code.
"""
from __future__ import annotations

import time

import mlx.core as mx
import numpy as np


def _to_torch_logits(logits):
    """MLX logits -> the BF16 torch tensor upstream's sampler expects."""
    import torch
    return torch.from_numpy(np.array(logits.astype(mx.float32))).to(torch.bfloat16)


def _prefill(model, ids):
    caches = model.new_caches()
    logits = model(mx.array([ids], dtype=mx.uint32), caches)
    mx.eval(logits)
    return logits, caches


def generate_tokens(model, prefix, sampling, seed, phase, negative=None, cfg_scale=1.0,
                    legacy_off=False, cancelled=None, on_token=None, **unused):
    """Signature-compatible replacement for ``yue2.sampling.generate_tokens``."""
    import torch
    from yue2.protocol import ABC_END, CONTEXT, MUSIC_END
    from yue2.sampling import distribution

    if len(prefix) + sampling.max_tokens > CONTEXT:
        raise ValueError("Prefix + requested generation budget exceeds 24576; no implicit truncation")
    if cfg_scale != 1 and negative is None:
        raise ValueError("CFG requires a negative prefix")
    if negative is not None and len(negative) + sampling.max_tokens > CONTEXT:
        raise ValueError("Negative prefix + generation budget exceeds context")
    if cancelled is not None and cancelled():
        raise InterruptedError("Cancelled before prefill")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    start = time.perf_counter()
    conditional_logits, positive_cache = _prefill(model, prefix)
    conditional = _to_torch_logits(conditional_logits[:, -1, :])
    unconditional, negative_cache = None, None
    if cfg_scale != 1.0:
        negative_logits, negative_cache = _prefill(model, negative)
        unconditional = _to_torch_logits(negative_logits[:, -1, :])
    prefill_seconds = time.perf_counter() - start

    history, first, eos = [], None, False
    end = ABC_END if phase == "abc" else MUSIC_END
    for step in range(sampling.max_tokens):
        if cancelled is not None and cancelled():
            raise InterruptedError(f"Cancelled during {phase}")
        logits = conditional if cfg_scale == 1.0 else unconditional + cfg_scale * (conditional - unconditional)
        scores = distribution(logits, sampling, history, step, phase, legacy_off)
        if sampling.temperature == 0:
            next_id = scores.argmax(-1, keepdim=True)
        else:
            next_id = torch.multinomial(scores.softmax(-1), 1, generator=generator)
        token = int(next_id.item())
        if first is None:
            first = time.perf_counter() - start
        if on_token is not None:
            on_token(phase, token)
        if token == end:
            eos = True
            break
        history.append(token)
        if step + 1 < sampling.max_tokens:
            current = mx.array([[token]], dtype=mx.uint32)
            conditional_logits = model(current, positive_cache)
            mx.eval(conditional_logits)
            conditional = _to_torch_logits(conditional_logits[:, -1, :])
            if negative_cache is not None:
                negative_logits = model(current, negative_cache)
                mx.eval(negative_logits)
                unconditional = _to_torch_logits(negative_logits[:, -1, :])

    seconds = time.perf_counter() - start
    count = len(history) + int(eos)
    quantization = model.config.get("quantization") or {}
    timing = {"seconds": seconds, "prefill_seconds": prefill_seconds,
              "ttft_seconds": first, "output_tokens": count, "content_tokens": len(history),
              "output_tps": count / seconds, "prefix_tokens": len(prefix),
              "cfg_branches": 1 if cfg_scale == 1 else 2,
              "execution": "mlx", "attention": "mlx_sdpa",
              "weights": f"mlx_q{quantization.get('bits')}" if quantization else "mlx_bf16"}
    return history, timing, not eos
