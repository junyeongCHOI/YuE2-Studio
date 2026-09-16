"""ABC score helpers: chord detection, chord stripping, duration.

cot="melody" does not remove chord symbols by itself, so a melody cover needs a
score that has none. Header fields also contain quoted strings (voice names),
which is why stripping only touches the tune body.
"""
from __future__ import annotations

import re

# A header/information field: a single letter, a colon, then the value.
HEADER = re.compile(r"^[A-Za-z]:")
# A chord symbol is a quoted string in the tune body, e.g. "Dm7" or "A7sus4".
# Annotations placed with ^ _ < > @ are text, not chords, and are left alone.
CHORD = re.compile(r'"(?![\^_<>@])([^"]*)"')


def _body_lines(abc):
    for index, line in enumerate(abc.splitlines()):
        stripped = line.lstrip()
        yield index, line, not (stripped.startswith("%") or HEADER.match(stripped))


def chord_symbols(abc):
    """Every chord symbol in the tune body, in order."""
    found = []
    for _, line, is_body in _body_lines(abc):
        if is_body:
            found += CHORD.findall(line)
    return found


def has_chords(abc):
    return bool(chord_symbols(abc or ""))


def strip_chords(abc):
    """Remove chord symbols from the tune body, leaving headers untouched."""
    out = []
    for _, line, is_body in _body_lines(abc):
        out.append(CHORD.sub("", line) if is_body else line)
    return "\n".join(out) + ("\n" if abc.endswith("\n") else "")


METER = re.compile(r"^M:\s*(\d+)\s*/\s*(\d+)", re.M)
TEMPO = re.compile(r"^Q:\s*(?:(\d+)\s*/\s*(\d+)\s*=\s*)?(\d+(?:\.\d+)?)", re.M)
VOICE = re.compile(r"^V:\s*(\S+)")


def duration_seconds(abc):
    """Playing time implied by the meter, tempo and bar count.

    Returns None if the header lacks M: or Q:.
    """
    abc = abc or ""
    meter = METER.search(abc)
    tempo = TEMPO.search(abc)
    if not meter or not tempo:
        return None
    beats, unit = int(meter.group(1)), int(meter.group(2))
    if not beats or not unit:
        return None
    # Q:1/4=108 counts quarter notes; a bare Q:108 means the same thing.
    reference = (int(tempo.group(1)) / int(tempo.group(2))) if tempo.group(1) else 0.25
    per_minute = float(tempo.group(3))
    if per_minute <= 0 or reference <= 0:
        return None

    # Voices are interleaved, so count bars per voice and take the longest.
    bars, voice = {}, None
    for _, line, is_body in _body_lines(abc):
        stripped = line.lstrip()
        marker = VOICE.match(stripped)
        if marker:
            voice = marker.group(1)
        elif is_body and voice is not None:
            bars[voice] = bars.get(voice, 0) + stripped.count("|")
    if not bars:
        return None

    quarters_per_bar = beats / unit / 0.25
    quarters = max(bars.values()) * quarters_per_bar
    return quarters / (per_minute * reference / 0.25) * 60


def analyse(abc):
    symbols = chord_symbols(abc or "")
    unique = sorted(set(symbols))
    return {"characters": len(abc or ""), "lines": len((abc or "").splitlines()),
            "duration_seconds": duration_seconds(abc),
            "has_chords": bool(symbols), "chord_count": len(symbols),
            "chords": unique[:24], "unique_chords": len(unique),
            "recommended_cot": "full" if symbols else "melody"}
