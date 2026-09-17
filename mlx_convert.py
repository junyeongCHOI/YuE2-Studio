"""Convert YuE2-3B to an 8-bit MLX checkpoint.

The released checkpoint is a Mixture-of-Transformers: each layer holds an AR
stack that token generation walks and a NAR stack that acoustic flow matching
walks. Both are converted, along with the flow-matching heads, so one directory
covers a whole song; the VAE decoder is separate and stays on PyTorch.

Token generation is memory-bandwidth bound, so packing its weights to 8 bits
makes it faster. Synthesis is not: it evaluates the whole latent sequence at
once, and quantizing that stack measured 3% faster while changing the audio it
produces (0.9991 waveform correlation against the same solver in BF16), so the
NAR half is left in BF16 unless asked.

    .venv/bin/python mlx_convert.py                      # -> models/YuE2-3B-mlx-8bit
    .venv/bin/python mlx_convert.py --ar-only            # token generation only
    .venv/bin/python mlx_convert.py --quantize-nar       # 8-bit synthesis too
    .venv/bin/python mlx_convert.py --bits 4 --out models/YuE2-3B-mlx-4bit
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx

import mlx_backend
from mlx_yue2 import is_quantized_module

# The NAR stack and the flow-matching heads; everything else is the AR half.
NAR_MARKERS = ("nar_", "llm2vae", "vae2llm", "time_embedder", "latent_pos_embed")
CONFIG_FIELDS = ("hidden_size", "num_hidden_layers", "num_attention_heads",
                 "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size",
                 "rms_norm_eps", "rope_theta", "max_position_embeddings",
                 "latent_dim", "max_latent_frames", "timestep_shift")
EXTRA_FILES = ("qwen.tiktoken", "LICENSE", "THIRD_PARTY_NOTICES.md",
               "generation_config.json", "yue2_generation_config.json")

CARD = """# YuE2-3B — MLX {bits}비트{suffix}

[m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) 를 MLX 형식으로 {bits}비트
양자화한 체크포인트. `mlx_convert.py` 가 만들었다.

## 무엇이 들어 있나

YuE2-3B 는 Mixture-of-Transformers 라 레이어마다 토큰을 만드는 AR 스택(`self_attn`/`mlp`)과
오디오를 합성하는 NAR 스택(`nar_*`)이 따로 있다.

| | |
|---|---|
| AR (악보·semantic 토큰) | `model.embed_tokens`, 28개 레이어의 AR 어텐션·MLP, `model.norm`, `lm_head` |
| NAR (플로우 매칭) | 28개 레이어의 NAR 어텐션·MLP, `vae2llm`, `llm2vae`, `time_embedder`, `latent_pos_embed` |
| 양자화 | {bits}비트 affine, 그룹 {group_size} — Linear/Embedding {quantized}개 |
| 원본 그대로 | {plain}개 텐서 ({plain_note}) |
| 제외 | {skipped}개 텐서 |
| 크기 | {original:.2f} GiB (bf16) → {converted:.2f} GiB ({ratio:.0%}) |

NAR 스택을 8비트로 만들어도 합성은 3% 밖에 빨라지지 않는다 — 잠재 시퀀스 전체를 한 번에
보는 계산이라 메모리 대역폭이 병목이 아니기 때문이다. 같은 솔버에서 8비트와 BF16 의
결과는 파형 상관도 0.9991 만큼 갈린다. 속도 이득이 없는데 근사를 더할 이유가 없어
기본값은 BF16 이고, 파일을 1.2GB 줄이려면 `--quantize-nar` 로 바꾼다.

VAE 디코더(`m-a-p/YuE2-Vae`)는 변환 대상이 아니다. 오디오 디코딩은 계속 PyTorch 로 돈다.

## 쓰는 법

```bash
./start --mlx                      # 대시보드 전체를 이 모델로
```

```python
import mlx_backend
mlx_backend.install()              # yue2 파이프라인의 AR·NAR 단계를 이 모델로
```

```python
import mlx_yue2
model, config = mlx_yue2.load("models/{name}")
```

정확도와 속도는 `mlx_verify.py` (토큰 생성) 와 `mlx_verify_nar.py` (합성) 가
PyTorch 와 비교해 출력한다.

## 출처

| | |
|---|---|
| 원본 | {source_model} |
| 경로 | `{source_path}` |
| `model.safetensors` sha256 | `{source_sha}` |
| 변환 | `mlx_convert.py`, mlx {mlx_version} |

## 라이선스

원본과 같은 **CC BY-NC 4.0**, 비상업적 용도에 한한다. `LICENSE` 와 `licenses/` 를
그대로 복사해 두었다.
"""


def module_path(key: str) -> str:
    return key.rsplit(".", 1)[0]


def is_nar(key: str) -> bool:
    return any(marker in key for marker in NAR_MARKERS)


def is_quantizable(key: str, skip) -> bool:
    return key.endswith(".weight") and is_quantized_module(module_path(key), skip)


def resolve_source(source: str | None) -> Path:
    if source:
        return Path(source)
    from yue2.storage import resolve_model
    return Path(resolve_model("m-a-p/YuE2-3B", local_files_only=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", help="YuE2-3B directory (default: the Hugging Face cache copy)")
    parser.add_argument("--out", default="models/YuE2-3B-mlx-8bit", help="destination directory")
    parser.add_argument("--bits", type=int, default=8, choices=(2, 3, 4, 6, 8))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--ar-only", action="store_true",
                        help="convert only the token-generation half (smaller; NAR stays on PyTorch)")
    parser.add_argument("--quantize-nar", action="store_true",
                        help="also quantize the synthesis stack (measured: no speed gain, more error)")
    parser.add_argument("--keep-bf16", nargs="*", default=[], metavar="MODULE",
                        help="module paths to leave unquantized, e.g. lm_head model.embed_tokens")
    parser.add_argument("--no-verify", action="store_true", help="skip the source checksum")
    args = parser.parse_args()

    source = resolve_source(args.source)
    destination = Path(args.out)
    checkpoint = source / "model.safetensors"
    if not checkpoint.is_file():
        raise SystemExit(f"No model.safetensors in {source}; run download_models.py first")
    if destination.exists() and any(destination.iterdir()):
        # A directory the loader cannot use is a leftover, not a model: replace
        # it rather than making the caller work out that it has to be deleted.
        if mlx_backend.available(destination):
            raise SystemExit(f"{destination} already exists and is not empty")
        print(f"replacing an unusable conversion at {destination}")
        shutil.rmtree(destination)

    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "yue2":
        raise SystemExit(f"Expected a yue2 checkpoint, found model_type={config.get('model_type')!r}")

    print(f"source      {source}")
    print(f"destination {destination}")
    print(f"contents    {'AR only' if args.ar_only else 'AR + NAR'}"
          + ("" if args.ar_only else f", NAR {'quantized' if args.quantize_nar else 'bf16'}"))
    print(f"quantization {args.bits}-bit, group {args.group_size}"
          + (f", bf16: {' '.join(args.keep_bf16)}" if args.keep_bf16 else ""))

    source_sha = None
    if not args.no_verify:
        from yue2.storage import sha256_file
        start = time.perf_counter()
        source_sha = sha256_file(checkpoint)
        print(f"source sha256 {source_sha[:16]}… ({time.perf_counter() - start:.1f}s)")

    weights = mx.load(str(checkpoint))
    keep = set(args.keep_bf16)
    if not args.quantize_nar and not args.ar_only:
        keep |= {f"model.layers.{i}.nar_{part}" for i in range(config["num_hidden_layers"])
                 for part in ("self_attn", "mlp")}
    skip = sorted(keep)
    converted: dict[str, mx.array] = {}
    skipped, quantized, plain = [], [], []
    original_bytes = converted_bytes = 0

    for key in sorted(weights):
        value = weights[key]
        if args.ar_only and is_nar(key):
            skipped.append(key)
            continue
        original_bytes += value.nbytes
        if is_quantizable(key, skip):
            w_q, scales, biases = mx.quantize(value, group_size=args.group_size, bits=args.bits)
            base = module_path(key)
            converted[key] = w_q
            converted[f"{base}.scales"] = scales
            converted[f"{base}.biases"] = biases
            converted_bytes += w_q.nbytes + scales.nbytes + biases.nbytes
            quantized.append(base)
        else:
            converted[key] = value
            converted_bytes += value.nbytes
            plain.append(key)
        mx.eval(list(converted.values())[-3:])

    # Everything is written into a staging directory and renamed at the end.
    # An interrupted conversion then leaves nothing at the destination, instead
    # of a half-written model that looks converted to ./start and to the loader
    # while mlx_convert refuses to overwrite it.
    staging = destination.parent / f".{destination.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    mx.save_safetensors(str(staging / "model.safetensors"), converted,
                        metadata={"format": "mlx"})

    out_config = {field: config[field] for field in CONFIG_FIELDS}
    out_config.update({
        "model_type": "yue2-mlx",
        "architectures": ["YuE2"],
        "dtype": "bfloat16",
        "quantization": {"group_size": args.group_size, "bits": args.bits, "skip": skip},
        "ar_only": args.ar_only,
        "source": {"model": "m-a-p/YuE2-3B", "path": str(source),
                   "model.safetensors.sha256": source_sha},
        "runtime": "mlx_yue2.py",
    })
    (staging / "config.json").write_text(json.dumps(out_config, indent=2) + "\n")
    for name in EXTRA_FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, staging / name)
    if (source / "licenses").is_dir():
        shutil.copytree(source / "licenses", staging / "licenses", dirs_exist_ok=True)

    (staging / "README.md").write_text(CARD.format(
        bits=args.bits, group_size=args.group_size, suffix=" (AR 전용)" if args.ar_only else "",
        quantized=len(quantized), plain=len(plain), skipped=len(skipped),
        plain_note=("RMSNorm 가중치, 플로우 매칭 헤드, 위치 임베딩 테이블"
                    if args.quantize_nar or args.ar_only else
                    "NAR 스택 전체, RMSNorm 가중치, 플로우 매칭 헤드, 위치 임베딩 테이블"),
        original=original_bytes / 2**30, converted=converted_bytes / 2**30,
        ratio=converted_bytes / original_bytes, name=destination.name,
        source_model="m-a-p/YuE2-3B", source_path=source, source_sha=source_sha or "(확인 생략)",
        mlx_version=mx.__version__))

    staging.rename(destination)

    print(f"\nquantized modules  {len(quantized)}")
    print(f"bf16 tensors       {len(plain)}")
    if skipped:
        print(f"tensors left behind {len(skipped)}")
    print(f"weights {original_bytes / 2**30:.2f} GiB bf16 -> {converted_bytes / 2**30:.2f} GiB "
          f"({converted_bytes / original_bytes:.1%})")
    print(f"written to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
