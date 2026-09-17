"""Local dashboard for YuE2 on Apple Silicon.

One process holds a warm pipeline and runs jobs one at a time; FastAPI serves
the queue, the library and the mastering chain to a single-page UI.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

import guidance
import mastering
import mlx_backend
import sampling_guard
import score as score_module
import store as store_module
import transcribe as transcribe_module

ROOT = Path(__file__).parent
LIBRARY = ROOT / "outputs" / "library"
LIBRARY.mkdir(parents=True, exist_ok=True)
STORE = store_module.Store(ROOT / "outputs" / "library.db")

# Installed at import so no code path can reach a decode without it.
sampling_guard.install()

# The converted MLX model runs the whole song except the VAE, and is the default
# when one is present. Patching happens before the pipeline is built; the
# dashboard can switch afterwards while the queue is idle.
def select_startup_backend():
    backend = os.environ.get("YUE2_BACKEND", "mlx").lower()
    model = os.environ.get("YUE2_MODEL") or None
    if backend == "torch":
        return "torch"
    if mlx_backend.use("mlx", model):
        return "mlx"
    where = mlx_backend.resolve(model)
    reason = ("mlx 가 설치되어 있지 않습니다" if not mlx_backend.mlx_importable()
              else f"변환된 모델이 없습니다 ({where})")
    print(f"! MLX 백엔드를 쓸 수 없습니다 — {reason}.\n"
          f"  .venv/bin/python mlx_convert.py 로 만들 수 있습니다. PyTorch 로 계속합니다.",
          flush=True)
    return "torch"


select_startup_backend()

JOB_ID = re.compile(r"^[0-9a-f]{6,32}$")

UPLOADS = ROOT / "outputs" / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_SCORE_CHARS = 200_000

# ---------------------------------------------------------------- job state

class Job:
    def __init__(self, identifier, request):
        # Path(library) / "" collapses to the library root, so a blank id would
        # make delete() rmtree every take.
        if not JOB_ID.match(str(identifier or "")):
            raise ValueError(f"invalid job id: {identifier!r}")
        self.id = identifier
        self.request = request
        self.status = "queued"
        self.created_at = time.time()
        self.started_at = None
        self.finished_at = None
        self.stage = None
        self.history = []
        self.error = None
        self.result = None
        self.mastering = None
        self.notes = ""
        self.favorite = False
        self.cancel_requested = False
        self.elapsed_override = None   # restored runs have no live start time
        self.lock = threading.Lock()

    @property
    def directory(self):
        return LIBRARY / self.id

    def begin_stage(self, label, total=None, unit=None):
        with self.lock:
            self.stage = {"label": label, "completed": 0, "total": total,
                          "unit": unit, "started_at": time.time()}

    def update_stage(self, completed=None, total=None):
        with self.lock:
            if self.stage is None:
                return
            if completed is not None:
                self.stage["completed"] = int(completed)
            if total is not None:
                self.stage["total"] = int(total)

    def advance_stage(self, count=1):
        with self.lock:
            if self.stage is not None:
                self.stage["completed"] += count

    def end_stage(self):
        with self.lock:
            if self.stage is None:
                return
            seconds = time.time() - self.stage["started_at"]
            if seconds >= 0.05 or self.stage["completed"]:
                self.history.append({"label": self.stage["label"], "seconds": seconds,
                                     "completed": self.stage["completed"]})
            self.stage = None

    def snapshot(self):
        with self.lock:
            if self.started_at:
                elapsed = (self.finished_at or time.time()) - self.started_at
            else:
                elapsed = self.elapsed_override or 0.0
            stage = dict(self.stage) if self.stage else None
            if stage is not None:
                stage["elapsed"] = time.time() - stage["started_at"]
            return {"id": self.id, "status": self.status, "request": self.request,
                    "created_at": self.created_at, "elapsed": elapsed,
                    "stage": stage, "history": list(self.history),
                    "error": self.error, "result": self.result,
                    "mastering": self.mastering, "notes": self.notes,
                    "favorite": self.favorite,
                    "cancel_requested": self.cancel_requested}

    def record(self):
        """Flatten the runtime state into a library row."""
        request, result = self.request, self.result or {}
        return {"id": self.id, "created_at": self.created_at, "status": self.status,
                "title": request.get("title", ""), "style": request.get("style", ""),
                "lyrics": request.get("lyrics", ""), "cot": request.get("cot", "full"),
                "seed": request.get("seed"), "cfg_scale": request.get("cfg_scale"),
                "max_tokens": request.get("max_tokens"),
                "ode_steps": request.get("ode_steps", 32), "preset": request.get("preset"),
                "mode": request.get("mode", "compose"), "source_job": request.get("source_job"),
                "source_kind": request.get("source_kind", "prompt"),
                "source_name": request.get("source_name"), "abc": request.get("abc"),
                "notes": self.notes, "favorite": self.favorite,
                "elapsed": self.snapshot()["elapsed"],
                "audio_seconds": result.get("audio_seconds"),
                "truncated": result.get("truncated"), "error": self.error,
                "timing": result.get("timing"), "history": self.history,
                "mastering": self.mastering, "request": request}

    @classmethod
    def from_record(cls, record):
        request = record.get("request") or {}
        # Columns win over the stored request blob: they carry the user's edits.
        for key in ("title", "style", "lyrics", "cot", "seed", "cfg_scale", "max_tokens",
                    "ode_steps", "preset", "mode", "source_job", "source_kind", "source_name", "abc"):
            if record.get(key) is not None:
                request[key] = record[key]
        job = cls(record["id"], request)
        job.status = "interrupted" if record["status"] in {"queued", "running"} else record["status"]
        job.created_at = record["created_at"]
        job.finished_at = record.get("updated_at") or record["created_at"]
        job.elapsed_override = record.get("elapsed")
        job.history = record.get("history") or []
        job.mastering = record.get("mastering")
        job.error = record.get("error")
        job.notes = record.get("notes") or ""
        job.favorite = bool(record.get("favorite"))
        if record.get("audio_seconds") is not None:
            job.result = {"audio_seconds": record["audio_seconds"],
                          "truncated": record.get("truncated"),
                          "timing": record.get("timing"), "sample_rate": 48000,
                          "ode_steps": record.get("ode_steps", 32)}
        return job

    def persist(self):
        STORE.upsert(self.record())


# ---------------------------------------------------------------- engine

class Engine:
    """Owns the pipeline and a single worker thread that drains the queue."""

    def __init__(self):
        self.jobs = {}
        self.order = []
        self.queue = []
        self.lock = threading.Lock()
        self.wakeup = threading.Condition(self.lock)
        self.pipeline = None
        self.pipeline_error = None
        self.loading = False
        self.current = None
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()
        self._restore()

    # -- library ---------------------------------------------------------

    def _restore(self):
        imported = store_module.import_json_library(STORE, LIBRARY)
        if imported:
            print(f"[library] imported {imported} take(s) from the pre-SQLite job.json files")
        for record in reversed(STORE.list(limit=1000)):
            job = Job.from_record(record)
            # A row left at queued/running by a restart has to be written back as
            # interrupted, or the listing keeps claiming it is live.
            if job.status != record["status"]:
                job.persist()
            with self.lock:
                self.jobs[job.id] = job
                self.order.append(job.id)

    # -- backend ---------------------------------------------------------

    def models(self):
        """Every backend the dashboard can switch to, current one marked."""
        current = mlx_backend.current()
        rows = [{"id": mlx_backend.TORCH_ID, "backend": "torch", "label": "PyTorch bf16 (원본)",
                 "stages": [], "bytes": None, "detail": "릴리스 가중치 그대로, MPS"}]
        for model in mlx_backend.discover():
            stages = "토큰 생성 + 합성" if "nar" in model["stages"] else "토큰 생성만"
            rows.append({**model,
                         "detail": f"{stages} · {model['bytes'] / 2**30:.1f}GB"})
        for row in rows:
            row["current"] = row["id"] == current["id"]
        return rows

    def set_backend(self, model_id):
        """Switch models between jobs; refuses while anything is in flight."""
        if model_id not in {row["id"] for row in self.models()}:
            raise KeyError(model_id)
        # The queue check and the patching have to be one atomic step: _run()
        # claims a job under this same lock, so a job submitted between the two
        # would otherwise start against half-swapped entry points. Patching is
        # a handful of attribute assignments; no weights are loaded here.
        with self.lock:
            if self.current is not None or self.queue or self.loading:
                raise ValueError("작업이 실행 중입니다. 끝난 뒤에 바꿀 수 있습니다")
            if model_id == mlx_backend.TORCH_ID:
                mlx_backend.use("torch")
            elif not mlx_backend.use("mlx", model_id):
                raise KeyError(model_id)
            # Drop whatever the previous backend had resident; the VAE is shared.
            if self.pipeline is not None:
                self.pipeline._model = None
        return self.models()

    # -- pipeline --------------------------------------------------------

    def _ensure_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline
        from yue2 import YuE2Pipeline
        self.loading = True
        try:
            self.pipeline = YuE2Pipeline.from_pretrained(
                "m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae", device="auto", progress=True)
            self.pipeline_error = None
        except Exception as exc:  # surfaced in /api/state
            self.pipeline_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.loading = False
        return self.pipeline

    @contextmanager
    def _tracking(self, job):
        """Mirror the pipeline's internal stage reporting into the job record."""
        pipe = self.pipeline
        original = pipe._status

        class Proxy:
            def __init__(self, inner):
                self._inner = inner

            def update(self, completed, total=None):
                job.update_stage(completed, total)
                return self._inner.update(completed, total=total)

            def advance(self, count=1):
                job.advance_stage(count)
                return self._inner.advance(count)

            def set_total(self, total):
                job.update_stage(total=total)
                return self._inner.set_total(total)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        @contextmanager
        def tracked(label, *, total=None, unit=None):
            job.begin_stage(label, total, unit)
            try:
                with original(label, total=total, unit=unit) as stage:
                    yield Proxy(stage)
            finally:
                job.end_stage()

        pipe._status = tracked
        try:
            yield
        finally:
            pipe._status = original

    # -- queue -----------------------------------------------------------

    def submit(self, request):
        job = Job(uuid.uuid4().hex[:12], request)
        with self.lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
            self.queue.append(job.id)
            self.wakeup.notify()
        job.persist()
        return job

    def cancel(self, job_id):
        job = self.jobs.get(job_id)
        if job is None:
            return None
        job.cancel_requested = True
        with self.lock:
            if job.id in self.queue:
                self.queue.remove(job.id)
                job.status = "cancelled"
                job.finished_at = time.time()
        job.persist()
        return job

    def delete(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return False
            if self.current is job:
                raise ValueError("Cancel the running job before deleting it")
            if job.id in self.queue:
                self.queue.remove(job.id)
            self.jobs.pop(job.id, None)
            if job.id in self.order:
                self.order.remove(job.id)
        # 8: the uploaded source lives outside the take's directory.
        upload = job.request.get("upload_path")
        if upload:
            try:
                parent = Path(upload).resolve().parent
                if parent.parent == UPLOADS.resolve():
                    shutil.rmtree(parent, ignore_errors=True)
            except OSError:
                pass
        shutil.rmtree(job.directory, ignore_errors=True)
        STORE.delete(job_id)
        return True

    def _run(self):
        while True:
            with self.wakeup:
                while not self.queue:
                    self.wakeup.wait()
                # Claim under the lock, or delete() can remove a job mid-flight.
                job = self.jobs[self.queue.pop(0)]
                self.current = job
            try:
                self._execute(job)
            except InterruptedError:
                # yue2 signals a honoured cancel by raising; that is not a failure.
                job.status = "cancelled"
                job.error = None
            except Exception:
                job.status = "cancelled" if job.cancel_requested else "failed"
                job.error = None if job.cancel_requested else traceback.format_exc(limit=4)
            finally:
                job.finished_at = time.time()
                job.persist()
                self.current = None

    @contextmanager
    def _ode_steps(self, pipe, steps):
        """Temporarily override the NAR step count; it is the audio-quality knob."""
        original = pipe.generation_config
        pipe.generation_config = dataclasses.replace(original, ode_steps=int(steps))
        try:
            yield
        finally:
            pipe.generation_config = original

    def _execute(self, job):
        job.status = "running"
        job.started_at = time.time()
        request = job.request

        if request.get("mode") == "transcribe":
            self._transcribe(job)
            return

        job.begin_stage("Loading model")
        try:
            pipe = self._ensure_pipeline()
        finally:
            job.end_stage()

        steps = int(request.get("ode_steps") or 32)
        sampling_guard.take_repairs()   # tally belongs to this run only
        with self._tracking(job), self._ode_steps(pipe, steps):
            if request.get("mode") == "resynth":
                song = self._resynthesize(pipe, job)
            else:
                song = self._compose(pipe, job)
        repairs = sampling_guard.take_repairs()

        job.directory.mkdir(parents=True, exist_ok=True)
        result = song.save_artifacts(job.directory)
        job.result = {"audio_seconds": result["audio_seconds"], "truncated": result["truncated"],
                      "timing": result["timing"], "sample_rate": result["sample_rate"],
                      "ode_steps": steps}
        if repairs["steps"]:
            job.result["logit_repairs"] = repairs

        preset = request.get("preset")
        if preset and preset != "none":
            job.begin_stage("Mastering")
            try:
                job.mastering = self._master(job, preset)
            finally:
                job.end_stage()

        job.status = "cancelled" if job.cancel_requested else "complete"

    def _transcribe(self, job):
        """Turn an uploaded recording into an ABC score.

        Shares the generation queue so the two never contend for the GPU.
        """
        request = job.request
        source = Path(request["upload_path"])
        job.begin_stage("Transcribing audio", unit="windows")

        def on_line(line):
            if line.startswith("Window "):
                position, _, total = line[7:].partition("/")
                try:
                    job.update_stage(int(position), int(total))
                except ValueError:
                    pass

        try:
            result = transcribe_module.transcribe(
                source, job.directory, melody_only=bool(request.get("melody_only")),
                max_seconds=request.get("max_seconds"), on_line=on_line)
        finally:
            job.end_stage()

        job.request["abc"] = result["abc"]
        job.result = {"seconds": result["seconds"], "source_name": request.get("source_name"),
                      "melody_only": bool(request.get("melody_only")),
                      "score": score_module.analyse(result["abc"])}
        job.status = "cancelled" if job.cancel_requested else "complete"

        follow_up = request.get("remix")
        if follow_up and not job.cancel_requested:
            job.result["remix_job"] = self._chain_remix(job, follow_up, result["abc"])

    def _chain_remix(self, source, follow_up, abc):
        """Queue the cover this transcription was made for.

        Strips chords for a melody cover rather than rejecting a request the
        caller has already waited minutes for.
        """
        payload = dict(follow_up)
        if payload.get("cot") == "melody" and score_module.has_chords(abc):
            abc = score_module.strip_chords(abc)
            source.result["chords_stripped"] = True
        payload.pop("strip_chords", None)
        payload.update({"abc": abc, "source_kind": "audio", "mode": "compose",
                        "source_job": source.id,
                        "source_name": source.request.get("source_name"),
                        "title": payload.get("title") or source.request.get("title", "")})
        return self.submit(payload).id

    def _compose(self, pipe, job):
        """Plan a score, generate semantic tokens, synthesise, decode."""
        from yue2.pipeline import SongResult
        from yue2.storage import identity

        request = job.request
        sampling = {"max_tokens": int(request["max_tokens"])} if request.get("max_tokens") else None
        song_request = pipe._request(style=request["style"], lyrics=request["lyrics"],
                                     cot=request.get("cot", "full"),
                                     seed=int(request.get("seed", 831001)),
                                     cfg_scale=request.get("cfg_scale"),
                                     abc=(request.get("abc") or None))
        score_scale = guidance.effective_scale(song_request.cot, request.get("score_scale"))
        config = pipe.effective_config(song_request, None, sampling)
        # Recorded only when it is in play, so takes made without it keep the
        # identity they would have had.
        if score_scale != guidance.DEFAULT_SCORE_SCALE:
            config["score_scale"] = score_scale
        request_identity = identity({"request": song_request.to_dict(), "config": config,
                                     "weights": pipe.weights})

        def cancelled():
            return job.cancel_requested

        start = time.perf_counter()
        plan = pipe.plan(request=song_request, cancelled=cancelled)
        semantic = (guidance.generate_semantic(pipe, plan, sampling=sampling,
                                               score_scale=score_scale, cancelled=cancelled)
                    if score_scale != guidance.DEFAULT_SCORE_SCALE
                    else pipe.generate_semantic(plan, sampling=sampling, cancelled=cancelled))
        nar_start = time.perf_counter()
        latents = pipe.synthesize(semantic, cancelled=cancelled)
        nar_seconds = time.perf_counter() - nar_start
        vae_start = time.perf_counter()
        audio = pipe.decode(latents)
        timing = {"abc": plan.timing, "semantic": semantic.timing, "nar_seconds": nar_seconds,
                  "vae_seconds": time.perf_counter() - vae_start,
                  "e2e_seconds": time.perf_counter() - start}
        return SongResult(audio, 48000, semantic, latents, config, pipe.weights,
                          timing, request_identity)

    def _extend_with_guidance(self, pipe, plan, existing, target, cancelled, score_scale):
        """The continuation, with the score's guidance branch in play."""
        from yue2.protocol import CODEC_OFFSET, CONTEXT, resolve_sampling

        request = plan.request
        prefixes, cfg_scale, score_scale = guidance.continuation_prefixes(
            pipe, plan, existing, score_scale=score_scale)
        room = CONTEXT - max(len(prefix) for prefix in prefixes)
        remaining = min(int(target) - len(existing), room)
        if remaining <= 0:
            return existing, {"output_tokens": len(existing), "reused": True}, room <= 0

        sampling = resolve_sampling({"max_tokens": remaining, "min_tokens": 0},
                                    pipe.generation_config.semantic)
        decoder, execution = guidance.decoder_for(pipe)
        with pipe._status("Extending song", unit="tokens") as status:
            observed = (lambda phase, token: status.advance()) if pipe.progress else None
            ids, timing, truncated = guidance.generate(
                decoder, prefixes, sampling, request.seed, "semantic", cfg_scale=cfg_scale,
                score_scale=score_scale, legacy_off=request.cot == "off",
                cancelled=cancelled, on_token=observed, execution=execution)
        timing = dict(timing, reused_tokens=len(existing))
        return existing + [int(t) - CODEC_OFFSET for t in ids], timing, truncated

    def _extend_semantic(self, pipe, plan, existing, target, cancelled,
                         score_scale=guidance.DEFAULT_SCORE_SCALE):
        """Continue the autoregressive decode from tokens that already exist.

        Appending the existing codec tokens to the prefix puts the model in the
        state an uninterrupted pass would have been in. Every guidance branch
        gets the same treatment, since each accumulates every sampled token.

        Existing tokens are reused verbatim; what follows is new music.
        """
        if score_scale != guidance.DEFAULT_SCORE_SCALE:
            return self._extend_with_guidance(pipe, plan, existing, target, cancelled, score_scale)
        from yue2.sampling import generate_tokens
        from yue2.protocol import CODEC_OFFSET, CONTEXT, negative_prefix, resolve_sampling

        request = plan.request
        prefix = list(plan.prefix) + [token + CODEC_OFFSET for token in existing]
        room = CONTEXT - len(prefix)
        remaining = min(int(target) - len(existing), room)
        if remaining <= 0:
            return existing, {"output_tokens": len(existing), "reused": True}, room <= 0

        # min_tokens would forbid the end token for another 200 steps; the song
        # may be nearly over already.
        sampling = resolve_sampling({"max_tokens": remaining, "min_tokens": 0},
                                    pipe.generation_config.semantic)
        negative = None
        if request.guidance != 1:
            negative = (negative_prefix(request, pipe.tokenizer, plan.abc_ids)
                        + [token + CODEC_OFFSET for token in existing])

        model = pipe._load_model()
        with pipe._status("Extending song", unit="tokens") as status:
            observed = (lambda phase, token: status.advance()) if pipe.progress else None
            ids, timing, truncated = generate_tokens(
                model, prefix, sampling, request.seed, "semantic",
                negative=negative, cfg_scale=request.guidance,
                legacy_off=request.cot == "off", cancelled=cancelled, on_token=observed,
                use_cuda_graph=pipe.backend != "torch-eager")
        timing = dict(timing, reused_tokens=len(existing))
        return existing + [int(t) - CODEC_OFFSET for t in ids], timing, truncated

    def _resynthesize(self, pipe, job):
        """Re-render a take at a different NAR step count, optionally longer.

        The semantic tokens carry the composition; only NAR/VAE turn them into
        audio. Reusing them skips the autoregressive half of the run. With
        extend_to the decode continues first.
        """
        from yue2.pipeline import SongResult, SymbolicPlan, SemanticResult
        from yue2.storage import identity

        source = self.jobs.get(job.request.get("source_job"))
        if source is None or not (source.directory / "semantic.npy").exists():
            raise FileNotFoundError("The source take is gone; generate it again")

        plan = SymbolicPlan.load(source.directory)
        existing = [int(t) for t in np.load(source.directory / "semantic.npy", allow_pickle=False)]

        def cancelled():
            return job.cancel_requested

        # The knob belongs to the take being continued, not to the resynth request.
        score_scale = guidance.effective_scale(plan.request.cot, source.request.get("score_scale"))
        target = job.request.get("extend_to")
        if target and int(target) > len(existing):
            tokens, semantic_timing, truncated = self._extend_semantic(
                pipe, plan, existing, int(target), cancelled, score_scale)
        else:
            tokens, semantic_timing, truncated = existing, {"output_tokens": len(existing),
                                                            "reused": True}, False
        semantic = SemanticResult(plan, tokens, semantic_timing, truncated)
        config = pipe.effective_config(plan.request)
        if score_scale != guidance.DEFAULT_SCORE_SCALE:
            config["score_scale"] = score_scale
        request_identity = identity({"request": plan.request.to_dict(), "config": config,
                                     "weights": pipe.weights})

        start = time.perf_counter()
        latents = pipe.synthesize(semantic, cancelled=cancelled)
        nar_seconds = time.perf_counter() - start
        vae_start = time.perf_counter()
        audio = pipe.decode(latents)
        timing = {"abc": plan.timing, "semantic": semantic_timing,
                  "nar_seconds": nar_seconds, "vae_seconds": time.perf_counter() - vae_start,
                  "e2e_seconds": time.perf_counter() - start}
        return SongResult(audio, 48000, semantic, latents, config, pipe.weights,
                          timing, request_identity)

    def _master(self, job, preset):
        source = job.directory / "audio.flac"
        destination = job.directory / "audio.mastered.flac"
        report = mastering.master_file(source, destination, preset)
        report["preset"] = preset
        (job.directory / "mastering.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=float) + "\n")
        return report

    def remaster(self, job_id, preset):
        job = self.jobs.get(job_id)
        if job is None or not (job.directory / "audio.flac").exists():
            raise KeyError(job_id)
        job.mastering = self._master(job, preset)
        job.request["preset"] = preset
        job.persist()
        return job.mastering

    def state(self):
        """Live queue only. Finished takes come from SQLite via /api/library."""
        with self.lock:
            current = self.current
            live = [current.id] if current is not None else []
            live += [i for i in self.queue if i != (current.id if current else None)]
            jobs = [self.jobs[i].snapshot() for i in live if i in self.jobs]
            queued = len(self.queue)
        device = str(self.pipeline.device) if self.pipeline else None
        return {"device": device, "logit_guard": sampling_guard.installed(),
                "backend": mlx_backend.current(), "models": self.models(),
                "pipeline_loaded": self.pipeline is not None,
                "pipeline_loading": self.loading, "pipeline_error": self.pipeline_error,
                "queued": queued, "presets": sorted(mastering.PRESETS),
                "running": current.id if current else None, "jobs": jobs}


engine = Engine()
app = FastAPI(title="YuE2 Studio")

ALLOWED_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}


@app.middleware("http")
async def loopback_only(request, call_next):
    """Reject requests not addressed to the loopback host.

    The server has no authentication because it is meant to be reachable only
    from this machine. A page in the browser can still send simple cross-origin
    POSTs, and a DNS rebinding attack can point a hostname at 127.0.0.1, so the
    Host header is checked rather than trusted.
    """
    host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
    if host and host not in {h.strip("[]") for h in ALLOWED_HOSTS}:
        return JSONResponse({"detail": "this server only answers on localhost"},
                            status_code=421)
    return await call_next(request)


class GenerateRequest(BaseModel):
    style: str = Field(min_length=1)
    lyrics: str = Field(min_length=1)
    abc: str | None = Field(default=None, max_length=MAX_SCORE_CHARS)
    strip_chords: bool = False        # drop chord symbols before a melody-only cover
    source_kind: str = "prompt"       # prompt | abc | audio
    source_name: str | None = None    # the file a transcribed score came from
    cot: str = "full"
    seed: int = 831001
    cfg_scale: float | None = None          # how hard tags/lyrics push
    score_scale: float | None = Field(default=None, ge=0, le=5)   # how hard the ABC does
    max_tokens: int | None = Field(default=None, ge=200, le=9000)
    ode_steps: int = Field(default=32, ge=2, le=64)
    preset: str = "streaming"
    title: str = ""


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "static" / "index.html").read_text()


@app.get("/api/models")
def models():
    """Backends this server can switch to without restarting."""
    return {"models": engine.models()}


class ModelSelect(BaseModel):
    id: str = Field(min_length=1, max_length=200)


@app.post("/api/models/select")
def select_model(body: ModelSelect):
    try:
        return {"models": engine.set_backend(body.id)}
    except ValueError as exc:      # something is running
        raise HTTPException(409, str(exc))
    except KeyError:
        raise HTTPException(404, "알 수 없는 모델입니다")


@app.get("/api/state")
def state():
    return engine.state()


class ScoreBody(BaseModel):
    abc: str = Field(default="", max_length=MAX_SCORE_CHARS)


@app.post("/api/abc/analyse")
def analyse_score(body: ScoreBody):
    """Report the chord symbols in a score and which cot mode suits it."""
    return score_module.analyse(body.abc)


@app.post("/api/abc/strip-chords")
def strip_chords(body: ScoreBody):
    """Remove chord symbols from the tune body, leaving header fields alone."""
    stripped = score_module.strip_chords(body.abc)
    return {"abc": stripped, "analysis": score_module.analyse(stripped)}


@app.post("/api/jobs")
def create(request: GenerateRequest):
    if request.cot not in {"full", "melody", "off"}:
        raise HTTPException(400, "cot must be full, melody or off")
    if request.preset != "none" and request.preset not in mastering.PRESETS:
        raise HTTPException(400, f"unknown preset {request.preset}")

    payload = request.model_dump()
    abc = payload.get("abc")
    if abc:
        # cot="melody" does not strip chord symbols itself, so a chord-annotated
        # score would feed the mode harmony it does not expect.
        if payload.pop("strip_chords", False):
            abc = score_module.strip_chords(abc)
            payload["abc"] = abc
        if request.cot == "melody" and score_module.has_chords(abc):
            raise HTTPException(400,
                'cot="melody" does not remove chord symbols. Either send '
                'strip_chords=true, or use cot="full" to keep the supplied harmony.')
    else:
        payload.pop("strip_chords", None)
    return engine.submit(payload).snapshot()


@app.post("/api/jobs/{job_id}/upgrade")
def upgrade(job_id: str, ode_steps: int = 32, preset: str | None = None,
            extend_to: int | None = None):
    """Re-render a take from its saved tokens, optionally continuing it.

    Without extend_to the performance is unchanged and only the NAR fidelity
    differs. With it, the decode continues first: the existing tokens are reused
    exactly, so the opening is the same music, though not the same waveform — NAR
    now renders it with the longer sequence as context.

    Re-running the request with a bigger cap instead would produce a different
    song, since the sampled tokens diverge from the first step.
    """
    source = engine.jobs.get(job_id)
    if source is None:
        raise HTTPException(404, "no such job")
    if not (source.directory / "semantic.npy").exists():
        raise HTTPException(409, "that take has no saved tokens to re-render")
    if not 2 <= ode_steps <= 64:
        raise HTTPException(400, "ode_steps must be between 2 and 64")
    if preset is not None and preset != "none" and preset not in mastering.PRESETS:
        raise HTTPException(400, f"unknown preset {preset}")
    if extend_to is not None and not 200 <= extend_to <= 9000:
        raise HTTPException(400, "extend_to must be between 200 and 9000")

    request = dict(source.request)
    request.update({"ode_steps": ode_steps, "source_job": job_id, "mode": "resynth",
                    "extend_to": extend_to,
                    "preset": preset or source.request.get("preset", "streaming")})
    label = source.request.get("title") or source.request.get("style", "")[:28]
    request["title"] = f"{label} · {'풀버전' if extend_to else f'ODE {ode_steps}'}"
    return engine.submit(request).snapshot()


@app.post("/api/jobs/{job_id}/retry")
def retry(job_id: str):
    """Queue the same request again — the usual next step after a failure."""
    source = engine.jobs.get(job_id)
    if source is None:
        raise HTTPException(404, "no such job")
    request = dict(source.request)
    if request.get("mode") != "resynth":
        request.pop("source_job", None)
    if request.get("mode") == "transcribe":
        upload = request.get("upload_path")
        if not upload or not Path(upload).is_file():
            raise HTTPException(409, "the uploaded audio is gone; upload it again")
    elif request.get("mode") == "resynth":
        parent = engine.jobs.get(request.get("source_job"))
        if parent is None or not (parent.directory / "semantic.npy").exists():
            raise HTTPException(409, "the take this was re-rendered from is gone")
    return engine.submit(request).snapshot()


@app.get("/api/transcription")
def transcription_status():
    return transcribe_module.requirements()


@app.post("/api/transcribe")
async def start_transcription(file: UploadFile = File(...), melody_only: bool = Form(False),
                              max_seconds: float | None = Form(None), title: str = Form(""),
                              remix: str | None = Form(None)):
    """Queue an audio file for transcription into an ABC score.

    remix is an optional JSON GenerateRequest; when present the transcription
    chains straight into it.
    """
    if not transcribe_module.available():
        raise HTTPException(503, "SheetSage2 is not installed; run ./setup_sheetsage.sh")
    follow_up = None
    if remix:
        try:
            follow_up = GenerateRequest(**{**json.loads(remix), "abc": None}).model_dump()
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, f"invalid remix request: {exc}")
        if follow_up["cot"] not in {"full", "melody", "off"}:
            raise HTTPException(400, "cot must be full, melody or off")
        if follow_up["preset"] != "none" and follow_up["preset"] not in mastering.PRESETS:
            raise HTTPException(400, f"unknown preset {follow_up['preset']}")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in transcribe_module.AUDIO_SUFFIXES:
        raise HTTPException(400, f"unsupported audio type {suffix or '(none)'}; "
                                 f"use one of {sorted(transcribe_module.AUDIO_SUFFIXES)}")
    destination = UPLOADS / uuid.uuid4().hex[:12]
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"source{suffix}"
    written = 0
    try:
        with path.open("wb") as sink:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413, f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
                sink.write(chunk)
        if not written:
            raise HTTPException(400, "empty file")
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    return engine.submit({"mode": "transcribe", "title": title or (file.filename or "transcription"),
                          "style": "", "lyrics": "", "source_kind": "audio",
                          "source_name": file.filename, "upload_path": str(path),
                          "melody_only": melody_only, "max_seconds": max_seconds,
                          "preset": "none", "remix": follow_up}).snapshot()


@app.get("/api/jobs/{job_id}/abc", response_class=PlainTextResponse)
def stored_abc(job_id: str):
    """The score a take used or produced, ready to paste into a remix."""
    job = engine.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job.request.get("abc"):
        return job.request["abc"]
    for name in ("score.abc",):
        path = job.directory / name
        if path.exists():
            return path.read_text()
    return ""


class JobEdit(BaseModel):
    title: str | None = None
    style: str | None = None
    lyrics: str | None = None
    notes: str | None = None
    favorite: bool | None = None


@app.get("/api/library")
def library(search: str | None = Query(default=None, max_length=200),
            status: str | None = Query(default=None, max_length=20),
            favorite: bool = False,
            limit: int = Query(default=200, ge=1, le=1000),
            offset: int = Query(default=0, ge=0)):
    """Search the stored takes. Returns rows straight from SQLite."""
    return {"total": STORE.count(search=search, status=status, favorite=favorite),
            "rows": STORE.list(search=search, status=status, favorite=favorite,
                               limit=limit, offset=offset)}


@app.patch("/api/jobs/{job_id}")
def edit(job_id: str, changes: JobEdit):
    """Rename a take, fix its style or lyrics text, add notes, star it."""
    fields = {k: v for k, v in changes.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(400, "nothing to change")
    record = STORE.update(job_id, **fields)
    if record is None:
        raise HTTPException(404, "no such job")
    job = engine.jobs.get(job_id)
    if job is not None:   # keep the live object in step with the row
        for key in ("title", "style", "lyrics"):
            if key in fields:
                job.request[key] = fields[key]
        job.notes = fields.get("notes", job.notes)
        job.favorite = bool(fields.get("favorite", job.favorite))
    return record


@app.get("/api/jobs/{job_id}")
def detail(job_id: str):
    job = engine.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job.snapshot()


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    job = engine.cancel(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job.snapshot()


@app.delete("/api/jobs/{job_id}")
def delete(job_id: str):
    try:
        if not engine.delete(job_id):
            raise HTTPException(404, "no such job")
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return {"deleted": job_id}


@app.post("/api/jobs/{job_id}/master")
def remaster(job_id: str, preset: str):
    if preset not in mastering.PRESETS:
        raise HTTPException(400, f"unknown preset {preset}")
    try:
        return engine.remaster(job_id, preset)
    except KeyError:
        raise HTTPException(404, "no such job, or it has no audio yet")


def uploaded_source(job):
    """The recording a transcription was made from.

    It lives in outputs/uploads, not in the take's directory, and is removed
    with the take. The path arrives back from SQLite, so it is confirmed to be
    inside the uploads tree before anything is served from it.
    """
    upload = job.request.get("upload_path")
    path = Path(upload).resolve() if upload else None
    if path is None or path.parent.parent != UPLOADS.resolve() or not path.is_file():
        raise HTTPException(404, "no uploaded source for this job")
    media_type = transcribe_module.AUDIO_MEDIA_TYPES.get(path.suffix.lower(),
                                                         "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=f"{job.id}-source{path.suffix}")


@app.get("/api/jobs/{job_id}/audio")
def audio(job_id: str, variant: str = "mastered"):
    job = engine.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if variant == "source":
        return uploaded_source(job)
    name = "audio.mastered.flac" if variant == "mastered" else "audio.flac"
    path = job.directory / name
    if not path.exists():
        path = job.directory / "audio.flac"
    if not path.exists():
        raise HTTPException(404, "no audio yet")
    return FileResponse(path, media_type="audio/flac", filename=f"{job_id}-{variant}.flac")


@app.get("/api/jobs/{job_id}/mastering")
def mastering_report(job_id: str):
    job = engine.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    path = job.directory / "mastering.json"
    if not path.exists():
        raise HTTPException(404, "not mastered")
    return JSONResponse(json.loads(path.read_text()))


@app.get("/api/presets")
def presets():
    return {name: json.loads(json.dumps(asdict(settings), default=float))
            for name, settings in mastering.PRESETS.items()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8710, log_level="info")
