"""Pre-download YuE2 weights using the same file allowlist the pipeline uses."""
from yue2.storage import MODEL_FILES, MODEL_LICENSES
from huggingface_hub import snapshot_download

patterns = (sorted(MODEL_FILES)
            + ["model-?????-of-?????.safetensors"]
            + ["licenses/" + n for n in sorted(MODEL_LICENSES)])

for repo in ("m-a-p/YuE2-3B", "m-a-p/YuE2-Vae"):
    print(f"==> {repo}", flush=True)
    path = snapshot_download(repo, allow_patterns=patterns)
    print(f"<== {repo} -> {path}", flush=True)
