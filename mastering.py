"""Programmatic mastering chain for YuE2 output. No models, just DSP.

Stages, in order:
  1. Spectral denoise      - STFT gating against an estimated noise floor
  2. Dynamic EQ            - cut-only bells that engage only when a band is hot
  3. Multiband compression - perfect-reconstruction band split + per-band comp
  4. Stereo correction     - bass mono, width control, polarity/correlation fix
  5. Limiter + LUFS        - ITU-R BS.1770 normalisation into a lookahead limiter

Every stage is optional and unity-safe: with default-off settings the chain
reconstructs its input sample-for-sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np
from scipy import signal, ndimage

CONTROL_DECIMATION = 64  # gain envelopes run at sr/64 (750 Hz at 48 kHz)


# ---------------------------------------------------------------- helpers

def _db(x, floor=1e-12):
    return 20.0 * np.log10(np.maximum(np.abs(x), floor))


def _lin(db):
    return 10.0 ** (np.asarray(db, dtype=np.float64) / 20.0)


def _jsonable(value):
    """Replace non-finite floats with None.

    Silence measures as -inf LUFS, and json.dumps would emit -Infinity, which
    JSON.parse rejects.
    """
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    return value


def _as_stereo(audio):
    """Return float64 (samples, channels) and whether the input was mono."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        return x[:, None], True
    if x.ndim != 2:
        raise ValueError("audio must be (samples,) or (samples, channels)")
    if x.shape[0] < x.shape[1]:  # (channels, samples) was passed
        x = x.T
    return x, False


def _lr4_lowpass(x, fc, sr):
    """Linkwitz-Riley 4th order lowpass: two cascaded 2nd order Butterworths."""
    sos = signal.butter(2, np.clip(fc / (sr / 2), 1e-6, 0.999), btype="low", output="sos")
    return signal.sosfilt(sos, signal.sosfilt(sos, x, axis=0), axis=0)


def _split_bands(x, crossovers, sr):
    """Complementary band split. The bands sum back to x exactly."""
    bands, rest = [], x
    for fc in crossovers:
        low = _lr4_lowpass(rest, fc, sr)
        bands.append(low)
        rest = rest - low
    bands.append(rest)
    return bands


def _envelope_db(x, decimation=CONTROL_DECIMATION):
    """Block-peak envelope in dB at the control rate, one value per block."""
    mono = np.max(np.abs(x), axis=1) if x.ndim == 2 else np.abs(x)
    blocks = len(mono) // decimation
    if blocks == 0:
        return _db(np.array([np.max(mono) if len(mono) else 0.0]))
    trimmed = mono[: blocks * decimation].reshape(blocks, decimation)
    peaks = trimmed.max(axis=1)
    if len(mono) > blocks * decimation:
        peaks = np.append(peaks, mono[blocks * decimation:].max())
    return _db(peaks)


def _smooth_gain(target_db, attack_ms, release_ms, sr, decimation=CONTROL_DECIMATION):
    """Attack/release smoothing of a gain-reduction curve at the control rate.

    Runs at sr/64, which keeps the recursion cheap while staying well above
    audio-rate modulation artefacts.
    """
    rate = sr / decimation
    attack = np.exp(-1.0 / max(attack_ms * 1e-3 * rate, 1e-9)) if attack_ms > 0 else 0.0
    release = np.exp(-1.0 / max(release_ms * 1e-3 * rate, 1e-9)) if release_ms > 0 else 0.0
    out = np.empty_like(target_db)
    state = 0.0
    for i, want in enumerate(target_db):
        coefficient = attack if want < state else release
        state = want + coefficient * (state - want)
        out[i] = state
    return out


def _to_sample_rate(control_db, samples, decimation=CONTROL_DECIMATION):
    """Linearly interpolate a control-rate curve up to the audio rate."""
    if len(control_db) == 1:
        return np.full(samples, control_db[0])
    positions = np.arange(len(control_db)) * decimation + (decimation - 1) / 2.0
    return np.interp(np.arange(samples), positions, control_db)


def _compress_curve(level_db, threshold_db, ratio, knee_db):
    """Static compressor curve -> gain reduction in dB (<= 0), with soft knee."""
    over = level_db - threshold_db
    reduction = np.zeros_like(over)
    if ratio <= 1.0:
        return reduction
    slope = 1.0 - 1.0 / ratio
    if knee_db > 0:
        inside = np.abs(over) <= knee_db / 2
        reduction = np.where(inside,
                             -slope * (over + knee_db / 2) ** 2 / (2 * knee_db),
                             np.where(over > 0, -slope * over, 0.0))
    else:
        reduction = np.where(over > 0, -slope * over, 0.0)
    return reduction


def _soft_clip(x, ceiling, knee):
    """Linear below knee * ceiling, tanh above it, asymptotic at the ceiling.

    Rounds peaks off instead of pulling them down with gain, so a competitive
    LUFS target does not force the limiter into many dB of pumping.
    """
    threshold = knee * ceiling
    magnitude = np.abs(x)
    over = magnitude > threshold
    if not over.any():
        return x
    span = max(ceiling - threshold, 1e-12)
    shaped = threshold + span * np.tanh((magnitude - threshold) / span)
    return np.where(over, np.sign(x) * shaped, x)


def true_peak_db(x, sr, oversample=4):
    """Inter-sample peak estimate via polyphase oversampling (dBTP).

    Resamples in float32; a peak magnitude does not need float64 precision and
    this is the most expensive measurement in the chain.
    """
    if len(x) == 0:
        return -np.inf
    up = signal.resample_poly(np.asarray(x, dtype=np.float32), oversample, 1, axis=0)
    return float(_db(np.max(np.abs(up))))


def correlation(x):
    """Stereo phase correlation in [-1, 1]. Mono returns 1.0."""
    if x.shape[1] < 2:
        return 1.0
    left, right = x[:, 0], x[:, 1]
    denominator = np.sqrt(np.mean(left ** 2) * np.mean(right ** 2))
    return float(np.mean(left * right) / denominator) if denominator > 0 else 1.0


def loudness_lufs(x, sr):
    """Integrated loudness, ITU-R BS.1770-4, via pyloudnorm."""
    import pyloudnorm
    if len(x) < int(0.4 * sr):  # meter needs at least one 400 ms block
        return float("-inf")
    meter = pyloudnorm.Meter(sr)
    return float(meter.integrated_loudness(x if x.shape[1] > 1 else x[:, 0]))


# ---------------------------------------------------------------- settings

@dataclass
class DenoiseSettings:
    enabled: bool = True
    floor_percentile: float = 10.0   # per-bin percentile over time taken as the noise floor
    sensitivity_db: float = 8.0      # a bin this far above its floor is treated as signal
    max_reduction_db: float = 14.0
    fft_size: int = 2048
    smooth_bins: int = 3
    smooth_frames: int = 5


@dataclass
class DynamicBand:
    name: str
    frequency: float
    bandwidth: float            # octaves
    threshold_offset_db: float  # relative to this band's own 75th percentile level
    ratio: float = 3.0
    max_cut_db: float = 6.0
    attack_ms: float = 15.0
    release_ms: float = 180.0


@dataclass
class DynamicEQSettings:
    enabled: bool = True
    reference_percentile: float = 75.0
    bands: list = field(default_factory=lambda: [
        DynamicBand("mud", 280.0, 1.2, 2.0, ratio=3.0, max_cut_db=5.0),
        DynamicBand("boxy", 550.0, 1.0, 2.5, ratio=2.5, max_cut_db=4.0),
        DynamicBand("harsh", 3200.0, 1.0, 2.0, ratio=3.0, max_cut_db=5.0,
                    attack_ms=8.0, release_ms=120.0),
        DynamicBand("sibilance", 7500.0, 1.2, 1.5, ratio=4.0, max_cut_db=6.0,
                    attack_ms=3.0, release_ms=80.0),
    ])


@dataclass
class CompressorBand:
    name: str
    threshold_offset_db: float  # relative to this band's own reference percentile level
    ratio: float
    attack_ms: float
    release_ms: float
    knee_db: float = 6.0
    makeup_db: float = 0.0


@dataclass
class MultibandSettings:
    enabled: bool = True
    crossovers: list = field(default_factory=lambda: [120.0, 800.0, 5000.0])
    reference_percentile: float = 80.0
    bands: list = field(default_factory=lambda: [
        CompressorBand("low", 1.0, 2.0, 20.0, 200.0, makeup_db=1.0),
        CompressorBand("low_mid", 1.0, 1.8, 15.0, 160.0, makeup_db=0.5),
        CompressorBand("high_mid", 1.0, 1.8, 10.0, 120.0, makeup_db=0.5),
        CompressorBand("high", 1.0, 1.8, 5.0, 90.0, makeup_db=0.5),
    ])


@dataclass
class StereoSettings:
    enabled: bool = True
    bass_mono_hz: float = 120.0
    width: float = 1.0             # side gain, 1.0 keeps the incoming width
    max_side_ratio: float = 1.4    # cap side RMS relative to mid RMS
    fix_polarity: bool = True      # flip one channel when correlation is strongly negative


@dataclass
class LimiterSettings:
    enabled: bool = True
    target_lufs: float = -14.0
    ceiling_dbtp: float = -1.0
    fast_ms: float = 4.0
    slow_ms: float = 80.0
    max_gain_db: float = 18.0      # refuse to push quiet material harder than this
    tolerance_lu: float = 0.1       # stop once the delivered LUFS is this close
    max_iterations: int = 8
    saturation_slope: float = 0.35  # below this LUFS-per-dB, more gain is only distortion
    soft_clip: bool = False         # tame peaks before limiting, for competitive targets
    soft_clip_knee: float = 0.7     # fraction of the ceiling below which clipping is linear


@dataclass
class MasteringSettings:
    denoise: DenoiseSettings = field(default_factory=DenoiseSettings)
    dynamic_eq: DynamicEQSettings = field(default_factory=DynamicEQSettings)
    multiband: MultibandSettings = field(default_factory=MultibandSettings)
    stereo: StereoSettings = field(default_factory=StereoSettings)
    limiter: LimiterSettings = field(default_factory=LimiterSettings)


PRESETS = {
    "streaming": MasteringSettings(),
    "loud": MasteringSettings(
        limiter=LimiterSettings(target_lufs=-9.0, ceiling_dbtp=-1.0, max_gain_db=26.0,
                                soft_clip=True),
        multiband=MultibandSettings(reference_percentile=78.0, bands=[
            CompressorBand("low", 0.0, 3.0, 15.0, 180.0, makeup_db=2.5),
            CompressorBand("low_mid", 0.0, 2.5, 12.0, 140.0, makeup_db=2.0),
            CompressorBand("high_mid", 0.0, 2.5, 8.0, 100.0, makeup_db=2.0),
            CompressorBand("high", 0.0, 2.5, 4.0, 80.0, makeup_db=2.0),
        ])),
    "gentle": MasteringSettings(
        denoise=DenoiseSettings(max_reduction_db=8.0, sensitivity_db=10.0),
        dynamic_eq=DynamicEQSettings(reference_percentile=85.0, bands=[
            DynamicBand("mud", 280.0, 1.2, 2.0, ratio=2.0, max_cut_db=3.0),
            DynamicBand("harsh", 3200.0, 1.0, 2.0, ratio=2.0, max_cut_db=3.0),
            DynamicBand("sibilance", 7500.0, 1.2, 1.5, ratio=2.5, max_cut_db=3.5,
                        attack_ms=3.0, release_ms=80.0),
        ]),
        multiband=MultibandSettings(reference_percentile=88.0, bands=[
            CompressorBand("low", 0.0, 1.8, 25.0, 250.0),
            CompressorBand("low_mid", 0.0, 1.6, 20.0, 200.0),
            CompressorBand("high_mid", 0.0, 1.6, 15.0, 150.0),
            CompressorBand("high", 0.0, 1.6, 8.0, 110.0),
        ]),
        limiter=LimiterSettings(target_lufs=-16.0, ceiling_dbtp=-1.5)),
    "transparent": MasteringSettings(
        denoise=DenoiseSettings(enabled=False),
        dynamic_eq=DynamicEQSettings(enabled=False),
        multiband=MultibandSettings(enabled=False),
        stereo=StereoSettings(enabled=False),
        limiter=LimiterSettings(target_lufs=-14.0, ceiling_dbtp=-1.0)),
}


# ---------------------------------------------------------------- stages

def spectral_denoise(x, sr, settings: DenoiseSettings):
    """Gate bins sitting at their own noise floor, leave everything else alone.

    The floor is a low percentile of each bin's magnitude over time; a bin more
    than sensitivity_db above it passes at unity. One mask is derived from the
    channel maximum and applied to both channels so the stereo image cannot drift.
    """
    if not settings.enabled or len(x) < settings.fft_size * 2:
        return x, {"applied": False}

    nperseg = settings.fft_size
    _, _, spectra = signal.stft(x, fs=sr, nperseg=nperseg,
                                noverlap=nperseg * 3 // 4, axis=0)
    magnitude = np.abs(spectra)
    reference = magnitude.max(axis=1) if magnitude.ndim == 3 else magnitude

    level_db = _db(reference)
    floor_db = np.percentile(level_db, settings.floor_percentile, axis=-1, keepdims=True)
    # 0 at the floor, 1 once the bin is sensitivity_db above it.
    openness = np.clip((level_db - floor_db) / max(settings.sensitivity_db, 1e-6), 0.0, 1.0)
    mask = _lin(-settings.max_reduction_db * (1.0 - openness))
    mask = ndimage.uniform_filter(mask, size=(settings.smooth_bins, settings.smooth_frames),
                                  mode="nearest")

    spectra = spectra * (mask[:, None, :] if spectra.ndim == 3 else mask)
    _, denoised = signal.istft(spectra, fs=sr, nperseg=nperseg,
                               noverlap=nperseg * 3 // 4, time_axis=-1, freq_axis=0)
    denoised = np.atleast_2d(denoised)
    if denoised.shape[0] < denoised.shape[1]:
        denoised = denoised.T
    denoised = denoised[: len(x)]
    if len(denoised) < len(x):
        denoised = np.pad(denoised, ((0, len(x) - len(denoised)), (0, 0)))

    gated = float(np.mean(openness < 1.0))
    residual = float(_db(np.sqrt(np.mean((x - denoised) ** 2)) + 1e-12)
                     - _db(np.sqrt(np.mean(x ** 2)) + 1e-12))
    return denoised, {"applied": True,
                      "noise_floor_db": float(np.mean(floor_db)),
                      "gated_bin_fraction": gated,
                      "removed_relative_db": residual}


def dynamic_eq(x, sr, settings: DynamicEQSettings):
    """Cut-only dynamic bells. Each band is extracted, measured, and subtracted."""
    if not settings.enabled or not settings.bands:
        return x, {"applied": False}

    out = x.copy()
    report = []
    for band in settings.bands:
        half = 2.0 ** (band.bandwidth / 2)
        low = np.clip(band.frequency / half / (sr / 2), 1e-6, 0.999)
        high = np.clip(band.frequency * half / (sr / 2), 1e-6, 0.999)
        if low >= high:
            continue
        sos = signal.butter(2, [low, high], btype="band", output="sos")
        extracted = signal.sosfilt(sos, out, axis=0)

        level_db = _envelope_db(extracted)
        threshold_db = float(np.percentile(level_db, settings.reference_percentile)
                             + band.threshold_offset_db)
        target_db = np.maximum(_compress_curve(level_db, threshold_db, band.ratio, 6.0),
                               -band.max_cut_db)
        smoothed = _smooth_gain(target_db, band.attack_ms, band.release_ms, sr)
        gain = _lin(_to_sample_rate(smoothed, len(out)))[:, None]

        out = out - extracted * (1.0 - gain)
        active = smoothed < -0.1
        report.append({"band": band.name, "frequency": band.frequency,
                       "threshold_db": threshold_db,
                       "max_cut_db": float(-smoothed.min()),
                       "active_fraction": float(np.mean(active)),
                       "mean_cut_when_active_db": float(-smoothed[active].mean()) if active.any() else 0.0})
    return out, {"applied": True, "bands": report}


def multiband_compress(x, sr, settings: MultibandSettings):
    """Compress complementary bands that sum back to the input exactly."""
    if not settings.enabled:
        return x, {"applied": False}
    if len(settings.bands) != len(settings.crossovers) + 1:
        raise ValueError("multiband needs exactly one more band than crossovers")

    bands = _split_bands(x, settings.crossovers, sr)
    out = np.zeros_like(x)
    report = []
    for signal_band, spec in zip(bands, settings.bands):
        level_db = _envelope_db(signal_band)
        threshold_db = float(np.percentile(level_db, settings.reference_percentile)
                             + spec.threshold_offset_db)
        target_db = _compress_curve(level_db, threshold_db, spec.ratio, spec.knee_db)
        smoothed = _smooth_gain(target_db, spec.attack_ms, spec.release_ms, sr)
        gain = _lin(_to_sample_rate(smoothed, len(x)) + spec.makeup_db)[:, None]
        out += signal_band * gain
        active = smoothed < -0.1
        report.append({"band": spec.name, "threshold_db": threshold_db,
                       "max_reduction_db": float(-smoothed.min()),
                       "active_fraction": float(np.mean(active)),
                       "mean_reduction_when_active_db": float(-smoothed[active].mean()) if active.any() else 0.0,
                       "makeup_db": spec.makeup_db})
    return out, {"applied": True, "crossovers": settings.crossovers, "bands": report}


def stereo_correct(x, sr, settings: StereoSettings):
    """Mid/side cleanup: mono bass, width cap, and optional polarity repair."""
    if not settings.enabled or x.shape[1] < 2:
        return x, {"applied": False, "channels": int(x.shape[1])}

    before = correlation(x)
    flipped = False
    if settings.fix_polarity and before < -0.5:
        x = x.copy()
        x[:, 1] = -x[:, 1]
        flipped = True

    mid = (x[:, 0] + x[:, 1]) / 2.0
    side = (x[:, 0] - x[:, 1]) / 2.0

    if settings.bass_mono_hz > 0:
        sos = signal.butter(2, np.clip(settings.bass_mono_hz / (sr / 2), 1e-6, 0.999),
                            btype="high", output="sos")
        side = signal.sosfilt(sos, signal.sosfilt(sos, side))

    side = side * settings.width

    mid_rms = float(np.sqrt(np.mean(mid ** 2)))
    side_rms = float(np.sqrt(np.mean(side ** 2)))
    narrowed = 1.0
    if mid_rms > 0 and side_rms > settings.max_side_ratio * mid_rms:
        narrowed = settings.max_side_ratio * mid_rms / side_rms
        side = side * narrowed

    out = np.stack([mid + side, mid - side], axis=1)
    return out, {"applied": True, "correlation_before": before,
                 "correlation_after": correlation(out), "polarity_flipped": flipped,
                 "side_narrowed_by": float(narrowed), "bass_mono_hz": settings.bass_mono_hz}


def limit_and_normalise(x, sr, settings: LimiterSettings):
    """Normalise to a LUFS target, then hold true peak under the ceiling.

    The gain curve is a lookahead minimum followed by a Hann smoothing of the
    same width. Each smoothed value averages minima over a window containing the
    sample itself, so the curve cannot exceed the per-sample target and the
    limiter cannot overshoot.

    Limiting costs loudness, so the normalise gain is re-solved against the
    limited result until the delivered LUFS settles.
    """
    if not settings.enabled:
        return x, {"applied": False}

    ceiling = float(_lin(settings.ceiling_dbtp))

    def apply_ceiling(audio):
        if settings.soft_clip:
            audio = _soft_clip(audio, ceiling, settings.soft_clip_knee)
        peak = np.max(np.abs(audio), axis=1)
        target_gain = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
        curves = []
        for milliseconds in (settings.fast_ms, settings.slow_ms):
            width = max(int(milliseconds * 1e-3 * sr), 1)
            held = ndimage.minimum_filter1d(target_gain, size=2 * width + 1, mode="nearest")
            window = np.hanning(2 * width + 1)
            window = window / window.sum()
            curves.append(np.convolve(held, window, mode="same"))
        envelope = np.minimum(curves[0], curves[1])[:, None]
        limited = audio * envelope
        # Inter-sample peaks can still exceed the ceiling; trim statically if so.
        achieved = true_peak_db(limited, sr)
        trim_db = min(0.0, settings.ceiling_dbtp - achieved)
        # A static trim shifts true peak by exactly that many dB, so there is no
        # need to oversample the trimmed signal again.
        return (limited * _lin(trim_db), float(-_db(envelope.min())), trim_db,
                achieved + trim_db)

    measured = loudness_lufs(x, sr)
    if not np.isfinite(measured):
        return x, {"applied": False, "reason": "programme too short to measure"}

    gain_db = float(np.clip(settings.target_lufs - measured,
                            -settings.max_gain_db, settings.max_gain_db))
    # Each dB of gain buys less than a dB of LUFS once the limiter engages, so
    # the step is scaled by the measured slope. A collapsed slope means the
    # programme is saturated; stop and keep the closest candidate.
    candidates, saturated, iterations = [], False, 0
    for iterations in range(1, settings.max_iterations + 1):
        out, reduction_db, trim_db, peak_dbtp = apply_ceiling(x * _lin(gain_db))
        delivered = loudness_lufs(out, sr)
        candidates.append({"gain_db": gain_db, "lufs": delivered,
                           "reduction_db": reduction_db, "trim_db": trim_db})
        error = settings.target_lufs - delivered
        if abs(error) <= settings.tolerance_lu:
            break
        slope = 1.0
        if len(candidates) >= 2 and candidates[-1]["gain_db"] != candidates[-2]["gain_db"]:
            slope = ((candidates[-1]["lufs"] - candidates[-2]["lufs"])
                     / (candidates[-1]["gain_db"] - candidates[-2]["gain_db"]))
            if slope < settings.saturation_slope:
                saturated = True
                break
            slope = float(min(slope, 1.0))
        step = float(np.clip(error / slope, -8.0, 8.0))
        proposed = float(np.clip(gain_db + step, -settings.max_gain_db, settings.max_gain_db))
        if proposed == gain_db:  # clamped, cannot get closer
            break
        gain_db = proposed

    best = min(candidates, key=lambda c: (abs(settings.target_lufs - c["lufs"]), c["gain_db"]))
    if best["gain_db"] != gain_db:
        gain_db = best["gain_db"]
        out, reduction_db, trim_db, peak_dbtp = apply_ceiling(x * _lin(gain_db))
    error = settings.target_lufs - best["lufs"]

    return out, {"applied": True, "input_lufs": measured, "normalise_db": gain_db,
                 "iterations": iterations,
                 "reached_target": bool(abs(error) <= settings.tolerance_lu),
                 "limited_by": "limiter saturation" if saturated else (
                     "max_gain_db" if abs(gain_db) >= settings.max_gain_db - 1e-9 else None),
                 "target_lufs": settings.target_lufs,
                 "max_reduction_db": reduction_db,
                 "static_trim_db": trim_db,
                 "output_lufs": best["lufs"],
                 "output_true_peak_dbtp": peak_dbtp,
                 "output_sample_peak_db": float(_db(np.max(np.abs(out))))}


# ---------------------------------------------------------------- chain

def master(audio, sr, settings: MasteringSettings | str = "streaming"):
    """Run the full chain. Returns (float32 audio, report dict)."""
    if isinstance(settings, str):
        if settings not in PRESETS:
            raise ValueError(f"Unknown preset {settings!r}; choose from {sorted(PRESETS)}")
        settings = PRESETS[settings]

    x, was_mono = _as_stereo(audio)
    report = {"sample_rate": sr, "duration_seconds": len(x) / sr,
              "input": {"lufs": loudness_lufs(x, sr),
                        "true_peak_dbtp": true_peak_db(x, sr),
                        "sample_peak_db": float(_db(np.max(np.abs(x)))) if len(x) else -np.inf,
                        "correlation": correlation(x)},
              "stages": {}}

    x, report["stages"]["denoise"] = spectral_denoise(x, sr, settings.denoise)
    x, report["stages"]["dynamic_eq"] = dynamic_eq(x, sr, settings.dynamic_eq)
    x, report["stages"]["multiband"] = multiband_compress(x, sr, settings.multiband)
    x, report["stages"]["stereo"] = stereo_correct(x, sr, settings.stereo)
    x, report["stages"]["limiter"] = limit_and_normalise(x, sr, settings.limiter)

    limiter = report["stages"]["limiter"]
    report["output"] = {
        # The limiter already measured both on exactly this signal.
        "lufs": limiter["output_lufs"] if limiter.get("applied") else loudness_lufs(x, sr),
        "true_peak_dbtp": limiter["output_true_peak_dbtp"] if limiter.get("applied")
                          else true_peak_db(x, sr),
        "sample_peak_db": float(_db(np.max(np.abs(x)))) if len(x) else -np.inf,
        "correlation": correlation(x)}
    if was_mono:
        x = x[:, 0]
    return x.astype(np.float32), _jsonable(report)


def master_file(source, destination, settings="streaming"):
    import soundfile as sf
    audio, sr = sf.read(str(source), always_2d=True, dtype="float64")
    processed, report = master(audio, sr, settings)
    subtype = "PCM_24" if str(destination).lower().endswith(".flac") else "FLOAT"
    sf.write(str(destination), processed, sr, subtype=subtype)
    report["source"], report["destination"] = str(source), str(destination)
    return report


def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Programmatic mastering for YuE2 output")
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--preset", default="streaming", choices=sorted(PRESETS))
    parser.add_argument("--target-lufs", type=float)
    parser.add_argument("--ceiling-dbtp", type=float)
    parser.add_argument("--report")
    arguments = parser.parse_args(argv)

    settings = PRESETS[arguments.preset]
    if arguments.target_lufs is not None or arguments.ceiling_dbtp is not None:
        import copy
        settings = copy.deepcopy(settings)
        if arguments.target_lufs is not None:
            settings.limiter.target_lufs = arguments.target_lufs
        if arguments.ceiling_dbtp is not None:
            settings.limiter.ceiling_dbtp = arguments.ceiling_dbtp

    report = master_file(arguments.source, arguments.destination, settings)
    text = json.dumps(report, indent=2, ensure_ascii=False, default=float)
    if arguments.report:
        open(arguments.report, "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
