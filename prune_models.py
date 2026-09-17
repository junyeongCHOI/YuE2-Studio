"""Delete original checkpoints that a conversion has replaced, from the Hugging Face cache.

    .venv/bin/python prune_models.py m-a-p/YuE2-3B m-a-p/MERT-v2-FullSong

./start runs this right after a conversion succeeds, unless --keep-originals.
Each comes back when something needs it: download_models.py or ./start --torch
fetches YuE2-3B again, and SheetSage2 fetches MERT-v2 on a PyTorch-only
transcription.
"""
import sys

from huggingface_hub import scan_cache_dir


def prune(repo_ids):
    cache = scan_cache_dir()
    for repo in cache.repos:
        if repo.repo_type != "model" or repo.repo_id not in repo_ids:
            continue
        strategy = cache.delete_revisions(*(revision.commit_hash for revision in repo.revisions))
        strategy.execute()
        print(f"  {repo.repo_id} 삭제 ({strategy.expected_freed_size_str} 확보)")


if __name__ == "__main__":
    prune(set(sys.argv[1:]))
