"""Acoustic flow matching on MLX -- the port of yue2/nar.py's CachedNAR.

The reference solver is already the efficient formulation: the AR prefix is
pushed through the AR stack once and its per-layer K/V kept, then every ODE
evaluation only touches the latent positions, attending over prefix + latents.
This mirrors it operation for operation -- same chunk cuts, same seeded CPU
noise draw (taken from ``yue2.nar.song_chunks``), same 32-step midpoint solver,
same BF16 state arithmetic -- so the only difference is where the matmuls run.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}


def _logit(t: float) -> float:
    """torch.logit(t).clamp(-20, 20) in float64, including logit(1) -> 20."""
    with np.errstate(divide="ignore"):
        return float(np.clip(np.log(np.float64(t) / (1.0 - np.float64(t))), -20.0, 20.0))


class CachedNAR:
    """One original acoustic chunk; the AR prefix K/V is invariant during the ODE."""

    def __init__(self, model, chunk):
        self.model, self.chunk = model, chunk
        self.dtype = DTYPES[model.config.get("dtype", "bfloat16")]
        noise = np.asarray(chunk.noise, dtype=np.float32)
        if noise.ndim != 2 or noise.shape[1] != 64 or len(noise) < 1:
            raise ValueError("Expected nonempty acoustic noise [frames,64]")
        if not np.isfinite(noise).all():
            raise ValueError("Acoustic noise contains non-finite values")
        self.noise = noise
        self.ar_length, self.nar_length = len(chunk.ar_tokens), len(noise) + 2
        if self.ar_length < 1 or min(chunk.ar_tokens) < 0 or max(chunk.ar_tokens) >= model.config["vocab_size"]:
            raise ValueError("AR prefix is empty or outside the model vocabulary")
        if self.ar_length + self.nar_length > model.config["max_position_embeddings"]:
            raise ValueError("Original acoustic chunk exceeds the model context")
        if chunk.nar_cond_end < 0:
            raise ValueError("nar_cond_end must be nonnegative")
        self.visible_length = min(chunk.nar_cond_end, self.ar_length) if chunk.nar_cond_end else self.ar_length
        local = mx.minimum(mx.arange(self.nar_length), model.config["max_latent_frames"] - 1)
        self.pos_emb = model.latent_pos_embed(local)[None]
        self.cache = []
        self._prefill()

    def _attend(self, attention, q, k, v, mask=None):
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=attention.scale, mask=mask)
        return out.transpose(0, 2, 1, 3).reshape(1, q.shape[2], -1)

    def _prefill(self):
        """Run the AR prefix through the AR stack, keeping each layer's K/V."""
        backbone = self.model.model
        ids = mx.array([list(self.chunk.ar_tokens)], dtype=mx.uint32)
        x = backbone.embed_tokens(ids)
        for layer in backbone.layers:
            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x))
            # Slice before eval so restricted visibility does not retain the
            # invisible codec tokens' storage for every layer.
            self.cache.append((k[:, :, :self.visible_length], v[:, :, :self.visible_length]))
            h = self._attend(layer.self_attn, q, k, v, mask="causal")
            x = x + layer.self_attn.o_proj(h)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        mx.eval(self.cache)

    def velocity(self, state, raw_t):
        """v_theta(x_t, t) over the latent positions of this chunk."""
        model = self.model
        if state.shape != (len(self.noise), 64):
            raise ValueError("ODE state shape changed")
        x_nar = mx.pad(state, [(1, 1), (0, 0)])
        shifted = model.shift_t(raw_t, self.dtype)
        x = model.vae2llm(x_nar[None])
        x = x + model.time_embedder(mx.broadcast_to(shifted, (self.nar_length,)), self.dtype)[None]
        x = x + self.pos_emb
        for layer, (ar_k, ar_v) in zip(model.model.layers, self.cache):
            attention = layer.nar_self_attn
            q, k, v = attention.project_qkv(layer.nar_input_layernorm(x), offset=self.ar_length)
            k = mx.concatenate([ar_k, k], axis=2)
            v = mx.concatenate([ar_v, v], axis=2)
            x = x + attention.o_proj(self._attend(attention, q, k, v))
            x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
        return model.llm2vae(model.model.norm(x))[0, 1:-1]

    def solve(self, steps=32, cancelled=None, on_progress=None):
        """32-step midpoint by default: two velocity evaluations per step."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        state = mx.array(self.noise).astype(self.dtype)
        dt = 1.0 / steps
        for step in range(steps):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            t = 1.0 - step * dt
            first = self.velocity(state, _logit(t))
            mid = state - first * (dt / 2)
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            state = state - self.velocity(mid, _logit(t - dt / 2)) * dt
            mx.eval(state)
            if on_progress is not None:
                on_progress(step + 1, int(steps))
        result = np.array(state.astype(mx.float32))
        if not np.isfinite(result).all():
            raise FloatingPointError("Acoustic flow matching produced non-finite latents")
        return result

    def close(self):
        self.cache.clear()
        self.pos_emb = None


def synthesize(model, prefix, codec, seed, steps=32, context=None, attention="sdpa",
               offload_ar=False, cancelled=None, query_chunk_size=None, on_progress=None):
    """Signature-compatible replacement for ``yue2.nar.synthesize``.

    ``attention``, ``offload_ar`` and ``query_chunk_size`` are PyTorch memory
    and kernel knobs; MLX needs none of them and they are accepted but ignored.
    """
    import torch
    from yue2.nar import song_chunks
    from yue2.protocol import CONTEXT

    chunks = song_chunks(prefix, codec, seed, CONTEXT if context is None else context)
    output = []
    for index, chunk in enumerate(chunks):
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled before acoustic prefill")
        engine = CachedNAR(model, chunk)
        try:
            progress = None
            if on_progress is not None:
                def progress(completed, total, index=index):
                    on_progress(index * total + completed, total * len(chunks))
            output.append(engine.solve(steps, cancelled, on_progress=progress))
        finally:
            engine.close()
        del engine
        mx.clear_cache()
    return torch.from_numpy(np.concatenate(output, axis=0))
