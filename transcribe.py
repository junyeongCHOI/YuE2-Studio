"""Audio to ABC score via SheetSage2.

SheetSage2 pins torch 2.8 / transformers 4.45 / numpy 1.x, which cannot coexist
with yue2's pins, so it lives in .venv-sheetsage and runs as a subprocess.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parent
PYTHON = ROOT / ".venv-sheetsage" / "bin" / "python"
INFER = ROOT / "sheetsage2" / "infer.py"
# The formats the uploader accepts, with the media type a browser needs to play
# them back: mimetypes guesses audio/mp4a-latm for .m4a and audio/x-flac for
# .flac, neither of which <audio> will touch.
AUDIO_MEDIA_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
                     ".m4a": "audio/mp4", ".aac": "audio/aac", ".ogg": "audio/ogg",
                     ".opus": "audio/ogg", ".aiff": "audio/aiff", ".aif": "audio/aiff"}
AUDIO_SUFFIXES = frozenset(AUDIO_MEDIA_TYPES)


def available():
    return PYTHON.is_file() and INFER.is_file()


def requirements():
    return {"python": str(PYTHON), "infer": str(INFER), "available": available(),
            "setup": "./setup_sheetsage.sh"}


def transcribe(audio, output_dir, *, melody_only=False, max_seconds=None,
               device="mps", dtype="bf16", timeout=3600, on_line=None,
               local_files_only=False):
    """Return {"abc": str, "seconds": float, "output_dir": str}.

    melody_only drops chord symbols. Raises RuntimeError with the output tail if
    no score is produced, TimeoutError if the deadline passes.
    """
    if not available():
        raise RuntimeError("SheetSage2 is not installed; run ./setup_sheetsage.sh")
    audio, output_dir = Path(audio), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # MERT-v2-FullSong is fetched on the first run, so offline stays optional.
    command = [str(PYTHON), str(INFER), str(audio), "--output", str(output_dir),
               "--device", device, "--dtype", dtype]
    if local_files_only:
        command.append("--local-files-only")
    if melody_only:
        command.append("--melody-only")
    if max_seconds:
        command += ["--max-seconds", str(float(max_seconds))]

    start = time.perf_counter()
    tail, timed_out = [], threading.Event()
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
        try:
            for line in process.stdout:
                line = line.rstrip()
                tail.append(line)
                del tail[:-40]
                if on_line is not None:
                    on_line(line)
            code = process.wait()
        finally:
            watchdog.cancel()

    if timed_out.is_set():
        raise TimeoutError(f"SheetSage2 exceeded {timeout:.0f}s and was stopped:\n"
                           + "\n".join(tail[-12:]))
    score = output_dir / "score.abc"
    if code != 0 or not score.is_file():
        raise RuntimeError("SheetSage2 could not transcribe this audio:\n" + "\n".join(tail[-12:]))
    return {"abc": score.read_text(), "seconds": time.perf_counter() - start,
            "output_dir": str(output_dir)}
