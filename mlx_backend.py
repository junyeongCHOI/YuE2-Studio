"""Route YuE2's token generation and acoustic synthesis through an MLX model.

``install()`` redirects the pipeline's AR stages (``yue2.sampling.generate_tokens``,
which the plan/semantic stages and the server's extend path both resolve at call
time) and, when the converted checkpoint carries the NAR stack, acoustic
synthesis (``yue2.nar.synthesize``) as well. It also stops the pipeline from
loading the 6.8GB BF16 torch model for work the converted checkpoint covers.

``uninstall()`` puts the originals back, so a running server can move between
converted models and the stock PyTorch path without restarting. The VAE decoder
is not converted and keeps running on PyTorch either way.

    import mlx_backend
    mlx_backend.use("mlx")                    # the default converted model
    mlx_backend.use("mlx", "models/YuE2-3B-mlx-4bit")
    mlx_backend.use("torch")                  # back to the released weights
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DEFAULT_MODEL_DIR = MODELS_DIR / "YuE2-3B-mlx-8bit"
TORCH_ID = "torch"
TORCH_REPO = "m-a-p/YuE2-3B"

_lock = threading.Lock()
_state = {"dir": None, "model": None, "installed": False, "stages": (), "originals": None}


def _config(model_dir):
    return json.loads((Path(model_dir) / "config.json").read_text())


def mlx_importable() -> bool:
    try:
        import mlx.core  # noqa: F401  -- the wheel is Apple Silicon only
    except ImportError:
        return False
    return True


def available(model_dir=None) -> bool:
    directory = Path(model_dir or DEFAULT_MODEL_DIR)
    return ((directory / "config.json").is_file()
            and (directory / "model.safetensors").is_file()
            and mlx_importable())


def self_contained(model_dir=None) -> bool:
    """Whether songs can be made from this conversion with the original deleted.

    The pipeline still needs the tokenizer, which the converter copies over; an
    AR-only conversion hands synthesis back to the BF16 torch model.
    """
    directory = Path(model_dir or DEFAULT_MODEL_DIR)
    if not available(directory) or not (directory / "qwen.tiktoken").is_file():
        return False
    try:
        return not _config(directory).get("ar_only", False)
    except (OSError, ValueError):
        return False


def torch_available() -> bool:
    """Whether the released BF16 weights are in the Hugging Face cache.

    A cache lookup only, so it is cheap enough for every state poll and never
    starts the 7GB download that asking snapshot_download would.
    """
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(TORCH_REPO, "model.safetensors"), str)


def describe(model_dir) -> dict:
    """One converted model, as the console and the dashboard show it."""
    directory = Path(model_dir)
    config = _config(directory)
    quantization = config.get("quantization") or {}
    stages = ["ar"] if config.get("ar_only") else ["ar", "nar"]
    bits = quantization.get("bits")
    return {
        "id": str(directory.relative_to(ROOT) if directory.is_relative_to(ROOT) else directory),
        "backend": "mlx",
        "path": str(directory),
        "bits": bits,
        "group_size": quantization.get("group_size"),
        "stages": stages,
        "bytes": (directory / "model.safetensors").stat().st_size,
        "label": f"MLX {bits}비트 ({'+'.join(s.upper() for s in stages)})",
    }


def discover(models_dir=None) -> list:
    """Every converted model under models/, newest conversion first."""
    root = Path(models_dir or MODELS_DIR)
    found = []
    if root.is_dir():
        for directory in sorted(root.iterdir()):
            if directory.name.startswith(".") or not available(directory):
                continue
            try:
                found.append(describe(directory))
            except (OSError, ValueError, KeyError):
                continue
    return found


def resolve(model_id):
    """Accept an id from discover(), a path, or None for the default."""
    if model_id in (None, "", "default"):
        return DEFAULT_MODEL_DIR
    candidate = Path(model_id)
    if not candidate.is_absolute():
        candidate = ROOT / model_id
    return candidate


def model():
    """Load on first use; the weights are memory-mapped, so this is cheap."""
    with _lock:
        if _state["model"] is None:
            import mlx_yue2
            _state["model"], _ = mlx_yue2.load(_state["dir"] or DEFAULT_MODEL_DIR)
        return _state["model"]


def release():
    """Drop the MLX weights, e.g. before the torch model is loaded for NAR."""
    with _lock:
        if _state["model"] is None:
            return
        _state["model"] = None
        import gc
        import mlx.core as mx
        # Arrays still reachable through a cycle would survive clear_cache().
        gc.collect()
        mx.clear_cache()


def loaded() -> bool:
    return _state["model"] is not None


def pipeline_dir():
    """The directory the yue2 pipeline should be built on, or None for the original.

    A self-contained conversion stands in for the released checkpoint entirely:
    its tokenizer feeds the pipeline and its weights are the ones on record.
    """
    directory = _state["dir"]
    if _state["installed"] and self_contained(directory):
        return Path(directory)
    return None


def _generate_tokens(_torch_model, prefix, sampling, seed, phase, **kwargs):
    import mlx_ar
    return mlx_ar.generate_tokens(model(), prefix, sampling, seed, phase, **kwargs)


def _synthesize(_torch_model, prefix, codec, seed, **kwargs):
    import mlx_nar
    return mlx_nar.synthesize(model(), prefix, codec, seed, **kwargs)


def install(model_dir=None) -> bool:
    """Patch the AR (and, when converted, NAR) entry points. Idempotent."""
    directory = resolve(model_dir)
    with _lock:
        if _state["installed"] and Path(_state["dir"]) == directory:
            return True
    if not available(directory):
        return False
    try:
        nar = not _config(directory).get("ar_only", False)
    except (OSError, ValueError):
        return False        # unreadable config: keep whatever is installed now
    uninstall()

    import yue2.nar as nar_module
    import yue2.pipeline as pipeline
    import yue2.sampling as sampling

    originals = {"sampling": sampling.generate_tokens, "pipeline": pipeline.generate_tokens,
                 "nar": nar_module.synthesize, "load_model": pipeline.YuE2Pipeline._load_model}

    # pipeline.py binds generate_tokens at import; server.py imports it per call.
    sampling.generate_tokens = _generate_tokens
    pipeline.generate_tokens = _generate_tokens
    if nar:
        nar_module.synthesize = _synthesize

    def _load_model(self, for_nar=False):
        if not for_nar:
            return None
        if nar:
            return model()          # synthesis is ours too; torch is never loaded
        release()                   # bound the peak before the BF16 model arrives
        return originals["load_model"](self, for_nar=for_nar)

    pipeline.YuE2Pipeline._load_model = _load_model
    with _lock:
        _state.update(dir=directory, installed=True, originals=originals,
                      stages=("ar", "nar") if nar else ("ar",))
    return True


def uninstall():
    """Restore the stock PyTorch entry points and drop the MLX weights."""
    with _lock:
        originals = _state["originals"]
        _state.update(installed=False, originals=None, stages=(), dir=None)
    if originals is not None:
        import yue2.nar as nar_module
        import yue2.pipeline as pipeline
        import yue2.sampling as sampling
        sampling.generate_tokens = originals["sampling"]
        pipeline.generate_tokens = originals["pipeline"]
        nar_module.synthesize = originals["nar"]
        pipeline.YuE2Pipeline._load_model = originals["load_model"]
    release()


def use(backend, model_dir=None) -> bool:
    """Switch backends. Returns False when the request could not be honoured."""
    if backend == TORCH_ID:
        uninstall()
        return True
    if backend != "mlx":
        raise ValueError("backend must be 'mlx' or 'torch'")
    return install(model_dir)


def installed() -> bool:
    return _state["installed"]


def stages() -> tuple:
    return _state["stages"]


def current() -> dict:
    """What the pipeline is wired to right now."""
    if not _state["installed"]:
        return {"backend": TORCH_ID, "id": TORCH_ID, "label": "PyTorch bf16 (원본)",
                "stages": [], "loaded": False}
    return {**describe(_state["dir"]), "loaded": _state["model"] is not None}


def status():
    directory = _state["dir"] or DEFAULT_MODEL_DIR
    return {"installed": _state["installed"], "loaded": _state["model"] is not None,
            "model_dir": str(directory), "available": available(directory),
            "stages": list(_state["stages"]), "current": current()}
