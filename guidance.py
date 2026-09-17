"""Classifier-free guidance over the two things a remix is conditioned on.

The released protocol puts the score in the prompt itself: ``token_prefixes``
lays out instruction, tags, lyrics, then the ABC, and the acoustic tokens are
generated after it. There is no weight on that score -- it is context, so the
model follows it as written. Upstream's ``cfg_scale`` does not loosen it
either: its negative branch carries *the same ABC*, so the score cancels and
only the tags/lyrics get amplified (``cfg_negative: same_instruction_and_exact_abc``).

Adding a second negative branch -- the same prompt with the score removed --
makes the score's contribution visible as a difference and therefore scalable:

    logits = P + (cfg_scale - 1)(P - N_text) + (score_scale - 1)(P - N_score)

    P        instruction + tags + lyrics + ABC          (what upstream sends)
    N_text   instruction + ABC                          (upstream's negative)
    N_score  instruction + tags + lyrics + empty ABC    (this module's)

score_scale 1 reproduces today's behaviour exactly, 0 ignores the score and
generates from tags and lyrics alone, and above 1 extrapolates past a literal
reading. Each knob that is not 1 costs one extra branch, decoded in lockstep.

N_score keeps the instruction and the empty ABC block the checkpoint uses when
no score is supplied (``cot="off"`` sends ``[ABC_START, ABC_END]``), so the
only thing that differs from P is the score's content.
"""
from __future__ import annotations

import time

from yue2.protocol import ABC_END, ABC_START, CONTEXT, EOD, MUSIC_END, MUSIC_START

DEFAULT_SCORE_SCALE = 1.0


def score_free_prefix(request, tokenizer):
    """The positive prompt with the score's content removed."""
    return [EOD] + tokenizer.encode(request.text()) + [ABC_START, ABC_END, MUSIC_START]


def branch_count(cfg_scale, score_scale) -> int:
    """Positive, plus one negative per knob that is off its neutral value."""
    return 1 + (cfg_scale != 1) + (score_scale != 1)


def combine(logits, cfg_scale, score_scale):
    """Merge the branches ``plan()`` asked for, in their listed order.

    With one negative this is upstream's own expression, arithmetic included,
    so a run that does not use the score knob is unchanged.
    """
    # Which branch is which follows from the scales, so a list that does not
    # match them would be read as the wrong conditioning rather than fail.
    if len(logits) != branch_count(cfg_scale, score_scale):
        raise ValueError(f"cfg_scale={cfg_scale} and score_scale={score_scale} need "
                         f"{branch_count(cfg_scale, score_scale)} branches, got {len(logits)}")
    positive = logits[0]
    if len(logits) == 1:
        return positive
    if score_scale == 1:
        negative = logits[1]
        return negative + cfg_scale * (positive - negative)
    if cfg_scale == 1:
        negative = logits[1]
        return negative + score_scale * (positive - negative)
    text, score = logits[1], logits[2]
    return positive + (cfg_scale - 1) * (positive - text) + (score_scale - 1) * (positive - score)


def effective_scale(cot, score_scale):
    """``cot="off"`` sends an empty score block, so N_score would equal P.

    Weighting a difference that is zero by construction costs a branch and
    changes nothing, so the knob is neutralised rather than silently paid for.
    """
    if score_scale is None or cot == "off":
        return DEFAULT_SCORE_SCALE
    return float(score_scale)


def plan(request, tokenizer, abc_ids, *, score_scale=DEFAULT_SCORE_SCALE):
    """Prefixes to decode in lockstep: positive first, then the negatives used."""
    from yue2.protocol import negative_prefix, token_prefixes

    score_scale = effective_scale(request.cot, score_scale)
    cfg_scale = request.guidance
    prefixes = [token_prefixes(request, tokenizer, abc_ids)]
    if cfg_scale != 1:
        prefixes.append(negative_prefix(request, tokenizer, abc_ids))
    if score_scale != 1:
        prefixes.append(score_free_prefix(request, tokenizer))
    return prefixes, cfg_scale, score_scale


# ── decoders ──────────────────────────────────────────────────────────────────


class TorchDecoder:
    """One bounded KV cache per branch, on the released PyTorch model."""

    def __init__(self, model):
        self.model = model
        self.caches = []
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

    def prefill(self, prefixes, max_tokens):
        import torch
        from yue2.modeling_yue2 import StaticKVCache

        config = self.model.config
        logits = []
        for prefix in prefixes:
            cache = StaticKVCache(num_layers=config.num_hidden_layers, batch_size=1,
                                  num_kv_heads=config.num_key_value_heads,
                                  max_seq_len=len(prefix) + max_tokens,
                                  head_dim=config.head_dim, dtype=self.dtype, device=self.device)
            output = self.model(torch.tensor([prefix], device=self.device),
                                past_key_values=cache, use_cache=True, logits_to_keep=1)
            self.caches.append(cache)
            logits.append(output.logits[:, -1, :])
        return logits

    def step(self, token):
        import torch
        current = torch.tensor([[token]], device=self.device)
        return [self.model(current, past_key_values=cache, use_cache=True,
                           logits_to_keep=1).logits[:, -1, :] for cache in self.caches]

    def close(self):
        self.caches.clear()


class MLXDecoder:
    """The same, on a converted MLX model; its logits come back as torch BF16."""

    def __init__(self, model):
        import torch
        self.model = model
        self.caches = []
        self.device = torch.device("cpu")

    def prefill(self, prefixes, max_tokens):
        import mlx.core as mx
        import mlx_ar
        logits = []
        for prefix in prefixes:
            caches = self.model.new_caches()
            out = self.model(mx.array([prefix], dtype=mx.uint32), caches)
            mx.eval(out)
            self.caches.append(caches)
            logits.append(mlx_ar._to_torch_logits(out[:, -1, :]))
        return logits

    def step(self, token):
        import mlx.core as mx
        import mlx_ar
        current = mx.array([[token]], dtype=mx.uint32)
        logits = []
        for caches in self.caches:
            out = self.model(current, caches)
            mx.eval(out)
            logits.append(mlx_ar._to_torch_logits(out[:, -1, :]))
        return logits

    def close(self):
        self.caches.clear()


# ── entry points ──────────────────────────────────────────────────────────────


def decoder_for(pipe):
    """Whichever backend the pipeline is wired to right now."""
    import mlx_backend
    if mlx_backend.installed():
        return MLXDecoder(mlx_backend.model()), "mlx"
    return TorchDecoder(pipe._load_model()), "eager"


def generate_semantic(pipe, symbolic, *, sampling=None, score_scale=DEFAULT_SCORE_SCALE,
                      cancelled=None, on_token=None):
    """``YuE2Pipeline.generate_semantic`` with the score's own guidance branch.

    Kept to the same checks, the same resolved sampling and the same progress
    reporting, so the only difference from the stock path is the extra branch.
    """
    from yue2.pipeline import SemanticResult
    from yue2.protocol import CODEC_OFFSET, resolve_sampling, token_prefixes

    request = symbolic.request
    if token_prefixes(request, pipe.tokenizer, symbolic.abc_ids) != symbolic.prefix:
        raise ValueError("Plan prefix disagrees with request/exact ABC IDs")
    sampling = resolve_sampling(sampling, pipe.generation_config.semantic)
    prefixes, cfg_scale, score_scale = plan(request, pipe.tokenizer, symbolic.abc_ids,
                                            score_scale=score_scale)
    decoder, execution = decoder_for(pipe)
    with pipe._status("Generating song", unit="tokens") as status:
        def observed(phase, token):
            status.advance()
            if on_token is not None:
                on_token(phase, token)
        ids, timing, truncated = generate(
            decoder, prefixes, sampling, request.seed, "semantic", cfg_scale=cfg_scale,
            score_scale=score_scale, legacy_off=request.cot == "off", cancelled=cancelled,
            on_token=observed if pipe.progress else on_token, execution=execution)
        if truncated:
            status.finish(status="truncated")
    return SemanticResult(symbolic, [int(token) - CODEC_OFFSET for token in ids], timing, truncated)


def continuation_prefixes(pipe, symbolic, existing, *, score_scale=DEFAULT_SCORE_SCALE):
    """The same branches, each carrying the codec tokens already generated.

    Every branch accumulated each sampled token during the original run, so a
    continuation has to hand them the same history.
    """
    from yue2.protocol import CODEC_OFFSET

    prefixes, cfg_scale, score_scale = plan(symbolic.request, pipe.tokenizer, symbolic.abc_ids,
                                            score_scale=score_scale)
    tail = [int(token) + CODEC_OFFSET for token in existing]
    return [prefix + tail for prefix in prefixes], cfg_scale, score_scale


# ── the shared decode loop ────────────────────────────────────────────────────


def generate(decoder, prefixes, sampling, seed, phase, *, cfg_scale=1.0,
             score_scale=DEFAULT_SCORE_SCALE, legacy_off=False, cancelled=None,
             on_token=None, execution="eager"):
    """Decode every branch in lockstep and sample from the combination.

    Sampling itself is upstream's: ``yue2.sampling.distribution`` and the same
    seeded multinomial, so only the logits reaching it are new.
    """
    import torch
    from yue2.sampling import distribution, synchronize

    for prefix in prefixes:
        if len(prefix) + sampling.max_tokens > CONTEXT:
            raise ValueError("Prefix + requested generation budget exceeds 24576")
    if cancelled is not None and cancelled():
        raise InterruptedError("Cancelled before prefill")

    rng_device = decoder.device if decoder.device.type in {"cpu", "cuda"} else torch.device("cpu")
    generator = torch.Generator(device=rng_device).manual_seed(seed)
    synchronize(decoder.device)
    start = time.perf_counter()
    try:
        branch_logits = decoder.prefill(prefixes, sampling.max_tokens)
        synchronize(decoder.device)
        prefill_seconds = time.perf_counter() - start

        history, first, eos = [], None, False
        end = ABC_END if phase == "abc" else MUSIC_END
        for step in range(sampling.max_tokens):
            if cancelled is not None and cancelled():
                raise InterruptedError(f"Cancelled during {phase}")
            scores = distribution(combine(branch_logits, cfg_scale, score_scale),
                                  sampling, history, step, phase, legacy_off)
            if sampling.temperature == 0:
                next_id = scores.argmax(-1, keepdim=True)
            else:
                probabilities = scores.softmax(-1)
                if decoder.device.type == "mps":
                    next_id = torch.multinomial(probabilities.cpu(), 1, generator=generator).to(decoder.device)
                else:
                    next_id = torch.multinomial(probabilities, 1, generator=generator)
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
                branch_logits = decoder.step(token)

        synchronize(decoder.device)
        seconds = time.perf_counter() - start
        count = len(history) + int(eos)
        return history, {"seconds": seconds, "prefill_seconds": prefill_seconds,
                         "ttft_seconds": first, "output_tokens": count,
                         "content_tokens": len(history), "output_tps": count / seconds,
                         "prefix_tokens": len(prefixes[0]), "cfg_branches": len(prefixes),
                         "cfg_scale": cfg_scale, "score_scale": score_scale,
                         "execution": execution}, not eos
    finally:
        decoder.close()
