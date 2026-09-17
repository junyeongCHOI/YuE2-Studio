"""Pre-download YuE2 weights using the same file allowlist the pipeline uses.

--vae-only skips the 7.26GB released checkpoint, for when a self-contained MLX
conversion already stands in for it.
"""
import sys

from yue2.storage import MODEL_FILES, MODEL_LICENSES
from huggingface_hub import snapshot_download

patterns = (sorted(MODEL_FILES)
            + ["model-?????-of-?????.safetensors"]
            + ["licenses/" + n for n in sorted(MODEL_LICENSES)])

repos = ["m-a-p/YuE2-Vae"] if "--vae-only" in sys.argv[1:] else ["m-a-p/YuE2-3B", "m-a-p/YuE2-Vae"]
for repo in repos:
    print(f"==> {repo}", flush=True)
    path = snapshot_download(repo, allow_patterns=patterns)
    print(f"<== {repo} -> {path}", flush=True)
