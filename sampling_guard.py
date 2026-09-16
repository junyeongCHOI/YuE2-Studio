"""Tolerate the non-finite logits MPS occasionally produces mid-decode.

A single NaN logit makes torch.multinomial reject the whole distribution and
abort the run. NaN and +inf are folded to -inf so those tokens simply cannot be
picked; the decode only stops if every candidate is gone.

Wraps yue2.sampling.distribution, which generate_tokens resolves at call time.
"""
from __future__ import annotations

import threading

import torch
import yue2.sampling as _sampling

_original = _sampling.distribution
_lock = threading.Lock()
_repairs = {"steps": 0, "tokens": 0}


def _guarded(logits, sampling, history, step, phase, legacy_off=False):
    scores = _original(logits, sampling, history, step, phase, legacy_off)
    # -inf marks a forbidden token and is expected; NaN and +inf are not.
    broken = torch.isnan(scores) | torch.isposinf(scores)
    count = int(broken.sum())
    if not count:
        return scores

    repaired = scores.masked_fill(broken, float("-inf"))
    if bool(torch.isneginf(repaired).all()):
        raise RuntimeError(
            f"Every candidate token went non-finite at {phase} step {step}. "
            f"This is a numerical fault on MPS, not a problem with the request; "
            f"try again.")
    with _lock:
        _repairs["steps"] += 1
        _repairs["tokens"] += count
    return repaired


def install():
    """Idempotent. Call once before any generation."""
    if _sampling.distribution is not _guarded:
        _sampling.distribution = _guarded


def installed():
    return _sampling.distribution is _guarded


def take_repairs():
    """Read and clear the repair tally, for reporting against one run."""
    with _lock:
        tally = dict(_repairs)
        _repairs.update(steps=0, tokens=0)
    return tally
