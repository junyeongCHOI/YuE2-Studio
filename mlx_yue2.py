"""YuE2-3B on MLX: the Mixture-of-Transformers layers and the converted loader.

Each released layer carries two complete stacks over one shared attention
operation -- an AR stack (``self_attn`` / ``mlp``) that token generation walks,
and a NAR stack (``nar_self_attn`` / ``nar_mlp``) that acoustic flow matching
walks. Both are defined here under the checkpoint's own parameter names so a
converted file loads by name; mlx_ar.py drives the first, mlx_nar.py the second.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class KVCache:
    """Growing KV cache for single-token decoding; mlx-lm's allocation strategy."""

    def __init__(self, step: int = 256):
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = step

    def update_and_fetch(self, keys, values):
        prev = self.offset
        count = keys.shape[2]
        if self.keys is None or prev + count > self.keys.shape[2]:
            B, H, _, D = keys.shape
            # Grow to a whole number of steps that covers every used slot; the
            # unused tail is dropped first so the result stays contiguous.
            capacity = self.step * math.ceil((prev + count) / self.step)
            pad_k = mx.zeros((B, H, capacity - prev, D), keys.dtype)
            pad_v = mx.zeros((B, H, capacity - prev, values.shape[3]), values.dtype)
            if self.keys is None:
                self.keys, self.values = pad_k, pad_v
            else:
                self.keys = mx.concatenate([self.keys[:, :, :prev], pad_k], axis=2)
                self.values = mx.concatenate([self.values[:, :, :prev], pad_v], axis=2)
        self.keys[:, :, prev:prev + count] = keys
        self.values[:, :, prev:prev + count] = values
        self.offset = prev + count
        return self.keys[:, :, :self.offset], self.values[:, :, :self.offset]


class Attention(nn.Module):
    """One attention stack. AR and NAR layers hold one of these each."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.rope_theta = config["rope_theta"]
        self.scale = self.head_dim ** -0.5
        hidden = config["hidden_size"]
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, config["rms_norm_eps"])
        self.k_norm = nn.RMSNorm(self.head_dim, config["rms_norm_eps"])

    def project_qkv(self, x, offset=0):
        """Project, per-head normalize and rotate. [B,T,H,D] -> [B,H,T,D]."""
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)
        rope = dict(traditional=False, base=self.rope_theta, scale=1.0, offset=offset)
        return mx.fast.rope(q, self.head_dim, **rope), mx.fast.rope(k, self.head_dim, **rope), v

    def __call__(self, x, mask=None, cache=None):
        B, T, _ = x.shape
        q, k, v = self.project_qkv(x, cache.offset if cache is not None else 0)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden, inter = config["hidden_size"], config["intermediate_size"]
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """AR stack, plus the NAR stack when the converted file carries it."""

    def __init__(self, config, with_nar: bool):
        super().__init__()
        eps, hidden = config["rms_norm_eps"], config["hidden_size"]
        self.input_layernorm = nn.RMSNorm(hidden, eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = nn.RMSNorm(hidden, eps)
        self.mlp = MLP(config)
        if with_nar:
            self.nar_input_layernorm = nn.RMSNorm(hidden, eps)
            self.nar_self_attn = Attention(config)
            self.nar_pre_mlp_layernorm = nn.RMSNorm(hidden, eps)
            self.nar_mlp = MLP(config)

    def __call__(self, x, mask=None, cache=None):
        x = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep -> MLP, under the checkpoint's Sequential names."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        # nn.Sequential in the checkpoint: index 0 Linear, 1 SiLU, 2 Linear.
        self.mlp = [nn.Linear(frequency_embedding_size, hidden_size), nn.Identity(),
                    nn.Linear(hidden_size, hidden_size)]

    def __call__(self, t, dtype):
        half = self.frequency_embedding_size // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        args = t.astype(mx.float32)[:, None] * freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(dtype)
        return self.mlp[2](nn.silu(self.mlp[0](emb)))


class Backbone(nn.Module):
    def __init__(self, config, with_nar: bool):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = [DecoderLayer(config, with_nar) for _ in range(config["num_hidden_layers"])]
        self.norm = nn.RMSNorm(config["hidden_size"], config["rms_norm_eps"])

    def __call__(self, ids, caches=None, mask=None):
        x = self.embed_tokens(ids)
        for layer, cache in zip(self.layers, caches or [None] * len(self.layers)):
            x = layer(x, mask, cache)
        return self.norm(x)


class LatentPositionEmbedding(nn.Module):
    """Non-learnable sinusoidal table, kept as a parameter so it loads by name."""

    def __init__(self, max_frames: int, hidden_size: int):
        super().__init__()
        self.pe = mx.zeros((max_frames, hidden_size))

    def __call__(self, position_ids):
        return self.pe[position_ids]


class YuE2(nn.Module):
    """``model.*`` + the AR head, plus the NAR heads when they are present."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        with_nar = not config.get("ar_only", False)
        self.with_nar = with_nar
        self.model = Backbone(config, with_nar)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        if with_nar:
            self.llm2vae = nn.Linear(config["hidden_size"], config["latent_dim"])
            self.vae2llm = nn.Linear(config["latent_dim"], config["hidden_size"])
            self.time_embedder = TimestepEmbedder(config["hidden_size"])
            self.latent_pos_embed = LatentPositionEmbedding(config["max_latent_frames"],
                                                            config["hidden_size"])

    def __call__(self, ids, caches=None, last_only=True):
        mask = "causal" if ids.shape[1] > 1 else None
        hidden = self.model(ids, caches, mask)
        if last_only:
            hidden = hidden[:, -1:, :]
        return self.lm_head(hidden)

    def new_caches(self):
        return [KVCache() for _ in self.model.layers]

    def shift_t(self, raw_t: float, dtype):
        """sigmoid then the configured shift, in the model dtype as upstream does."""
        t_sig = mx.sigmoid(mx.array(raw_t).astype(dtype))
        shift = self.config.get("timestep_shift", 1.0)
        return shift * t_sig / (1 + (shift - 1) * t_sig)


# The modules mlx_convert.py packs. Everything else -- RMSNorm weights, the
# flow-matching heads, the latent position table -- stays in bf16, so the rule
# lives here and the converter imports it rather than restating it.
QUANTIZED_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def is_quantized_module(path: str, skip=()) -> bool:
    if path in skip or any(path.startswith(f"{entry}.") for entry in skip):
        return False
    if path in ("lm_head", "model.embed_tokens"):
        return True
    return path.startswith("model.layers.") and path.rsplit(".", 1)[-1] in QUANTIZED_LEAVES


def quantization_predicate(quantization):
    """Reproduce the converter's decision for every module in the tree."""
    skip = tuple(quantization.get("skip", ()))

    def predicate(path, module):
        return hasattr(module, "to_quantized") and is_quantized_module(path, skip)

    return predicate


def load(model_dir, *, lazy=False):
    """Load a converted MLX model directory produced by mlx_convert.py."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    model = YuE2(config)
    quantization = config.get("quantization")
    if quantization:
        nn.quantize(model, group_size=quantization["group_size"], bits=quantization["bits"],
                    class_predicate=quantization_predicate(quantization))
    model.load_weights(str(model_dir / "model.safetensors"))
    model.eval()
    if not lazy:
        mx.eval(model.parameters())
    return model, config
