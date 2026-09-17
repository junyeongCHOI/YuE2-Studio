"""Audio to ABC score via SheetSage2.

SheetSage2 pins torch 2.8 / transformers 4.45 / numpy 1.x, which cannot coexist
with yue2's pins, so it lives in .venv-sheetsage and runs as a subprocess.

Where one has been converted, its audio encoder runs on MLX and its decoder on
the CPU (mlx_sheetsage.py, sheetsage_runner.py); otherwise the whole model runs
in PyTorch on default_device().
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parent
PYTHON = ROOT / ".venv-sheetsage" / "bin" / "python"
INFER = ROOT / "sheetsage2" / "infer.py"
RUNNER = ROOT / "sheetsage_runner.py"
MLX_ENCODER = ROOT / "models" / "SheetSage2-encoder-mlx"
# The formats the uploader accepts, with the media type a browser needs to play
# them back: mimetypes guesses audio/mp4a-latm for .m4a and audio/x-flac for
# .flac, neither of which <audio> will touch.
AUDIO_MEDIA_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
                     ".m4a": "audio/mp4", ".aac": "audio/aac", ".ogg": "audio/ogg",
                     ".opus": "audio/ogg", ".aiff": "audio/aiff", ".aif": "audio/aiff"}
AUDIO_SUFFIXES = frozenset(AUDIO_MEDIA_TYPES)


def available():
    return PYTHON.is_file() and INFER.is_file()


def mlx_encoder():
    """Whether the encoder can run on MLX: converted, and mlx installed beside SheetSage2.

    On an 8GB M2 this took a 253 s song from 88 s to 61 s and its peak from
    2.48 GB to 2.36 GB, with the same ABC as the all-PyTorch FP32 run.
    """
    return ((MLX_ENCODER / "encoder.safetensors").is_file()
            and any((PYTHON.parent.parent / "lib").glob("python3*/site-packages/mlx")))


def default_device():
    """Where SheetSage2 runs without the MLX encoder: the GPU only with memory to spare.

    Every window is 300 s (short clips are padded) and MERT attends over all
    7500 frames of it. On MPS that is one 3.35 GiB attention matrix in FP32 --
    `--dtype bf16` autocasts on CUDA only -- and on an 8GB M2 the process
    reached 11 GB and paged for ten minutes without finishing a 15 s clip. The
    CPU kernel never builds that matrix: the same clip took 41 s at a 2.5 GB
    peak and a 253 s song 88 s, in FP32, so the score is the reference one.
    Loading BF16 weights on MPS fits a clip but garbles the decoder's output,
    and with the decoder kept in FP32 a full song still passed 5.4 GB.

    The dashboard's earlier timings (M5 / 32GB) were taken on MPS, so that is
    where the GPU is kept; below it, the CPU is the path known to fit.
    """
    try:
        memory = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                    text=True, timeout=5).stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return "cpu"
    return "mps" if memory >= 32 * 2**30 else "cpu"


DEVICE = default_device()


def requirements():
    return {"python": str(PYTHON), "infer": str(INFER), "available": available(),
            "setup": "./setup_sheetsage.sh", "device": "mlx" if mlx_encoder() else DEVICE}


def transcribe(audio, output_dir, *, melody_only=False, max_seconds=None,
               device=None, dtype="fp32", timeout=3600, on_line=None,
               local_files_only=False, cancelled=None):
    """Return {"abc": str, "seconds": float, "output_dir": str}.

    melody_only drops chord symbols. Raises RuntimeError with the output tail if
    no score is produced, TimeoutError if the deadline passes, InterruptedError
    once cancelled() turns true -- the process is killed then, which is what
    hands its models' memory back.
    """
    if not available():
        raise RuntimeError("SheetSage2 is not installed; run ./setup_sheetsage.sh")
    audio, output_dir = Path(audio), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # MERT-v2-FullSong is fetched on the first run, so offline stays optional.
    if device is None and mlx_encoder():
        command = [str(PYTHON), str(RUNNER), str(audio), "--output", str(output_dir),
                   "--device", "cpu", "--dtype", "fp32", "--mlx-encoder", str(MLX_ENCODER)]
    else:
        command = [str(PYTHON), str(INFER), str(audio), "--output", str(output_dir),
                   "--device", device or DEVICE, "--dtype", dtype]
    if local_files_only:
        command.append("--local-files-only")
    if melody_only:
        command.append("--melody-only")
    if max_seconds:
        command += ["--max-seconds", str(float(max_seconds))]

    start = time.perf_counter()
    tail, timed_out, stopped, done = [], threading.Event(), threading.Event(), threading.Event()
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        # Reading to EOF blocks forever on a silent hang, so the deadline has to
        # kill the process rather than guard process.wait().
        def expire():
            timed_out.set()
            process.kill()

        watchdog = threading.Timer(timeout, expire)
        watchdog.daemon = True
        watchdog.start()

        # A window can run for minutes without printing, so cancellation is
        # polled rather than checked between lines.
        def watch_cancel():
            while not done.wait(0.5):
                if cancelled():
                    stopped.set()
                    process.kill()
                    return

        if cancelled is not None:
            threading.Thread(target=watch_cancel, daemon=True).start()
        try:
            for line in process.stdout:
                line = line.rstrip()
                tail.append(line)
                del tail[:-40]
                if on_line is not None:
                    on_line(line)
            code = process.wait()
        finally:
            done.set()
            watchdog.cancel()

    if stopped.is_set():
        raise InterruptedError("Transcription cancelled")
    if timed_out.is_set():
        raise TimeoutError(f"SheetSage2 exceeded {timeout:.0f}s and was stopped:\n"
                           + "\n".join(tail[-12:]))
    score = output_dir / "score.abc"
    if code != 0 or not score.is_file():
        raise RuntimeError("SheetSage2 could not transcribe this audio:\n" + "\n".join(tail[-12:]))
    return {"abc": score.read_text(), "seconds": time.perf_counter() - start,
            "output_dir": str(output_dir)}
