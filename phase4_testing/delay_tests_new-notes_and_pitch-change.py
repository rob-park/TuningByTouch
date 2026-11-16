import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import butter, hilbert, sosfiltfilt


RUN_TAG = "fw1_contact"
ANALYSIS_VERSION = "v5"
FIRMWARE_VERSION = "settle=2"

MIC_DEV = 1
SPK_DEV = 3


CONDITIONS = [
    ("A4_20c", "pitch",         440.00, 20.0, 10),
    ("G3_20c", "pitch",         196.00, 20.0, 10),
    ("A6_20c", "pitch",        1760.00, 20.0, 10),
    ("A4_6c",  "pitch_report",  440.00,  6.0,  5),
    ("A4_0c",  "control",       440.00,  0.0,  4),
    ("G3_0c",  "control",       196.00,  0.0,  3),
    ("A6_0c",  "control",      1760.00,  0.0,  3),
]

SHUFFLE_SEED = 20261116
INTER_TRIAL_GAP = 2.0


SR = 48000
DEVICE_CONFIG = (MIC_DEV, SPK_DEV)
PRE_PAD = 0.5
DUR1 = 1.5
DUR2 = 1.5
POST_PAD = 1.0
FADE = 0.01


MOTOR_LOW = 1180.0
MOTOR_HIGH = 1250.0

WARMUP_CALLS = 3
WARMUP_LEN = 0.5


TONE_HALF_BW = 50.0
TONE_HALF_BW_LOW = 12.0
LOW_FREQ_CUTOFF = 250.0
TONE_K_LOW = 2.0
TONE_K_HIGH = 4.0
TONE_REL_FRAC = 0.25
TONE_MIN_SUSTAIN = 0.020
TONE_MIN_SNR_DB = 10.0
TONE_NOISE_WIN = (0.10, 0.40)
TONE_SEARCH_WIN = (0.55, 1.30)
TONE_VALID_WIN = (0.60, 1.20)
TONE_LEVEL_WIN = (0.30, 1.30)


MOTOR_REL_FRAC = 0.15
MOTOR_K = 3.0
MOTOR_MIN_SUSTAIN = 0.005
MOTOR_BASE_WIN = (-0.60, -0.10)
MOTOR_PRECHECK_WIN = (-0.30, -0.05)
MOTOR_PRE_SUSTAIN = 0.020
MOTOR_SEARCH_LEN = 0.50
MOTOR_LEVEL_PCT = 90
MOTOR_MIN_SNR_DB = 12.0
FALSE_TRIG_MIN = 0.010
SCAN_END_GUARD = 0.05
ONSET_IGNORE = 0.010
NOTE_OFF_WIN = 0.25

CLIP_LEVEL = 0.99
TARGET_MS = 100.0

STATUS_DESCRIPTIONS = {
    "OK": "Tone and motor onset detected; latency is valid.",
    "REC_ERROR": "The audio device failed to return a complete recording.",
    "MISSING_WAV": "The expected recording was not found during reanalysis.",
    "CLIPPED": "The microphone signal clipped; timing may be unreliable.",
    "NO_TONE": "No sustained tone onset was found in the search window.",
    "LOW_TONE_SNR": "Tone signal-to-noise ratio was below 10 dB.",
    "TONE_OUT_OF_WINDOW": "Tone onset was outside the expected 0.6–1.2 s window.",
    "ALREADY_BUZZING": "The motor was already above the session gate before the pitch change.",
    "NO_MOTOR": "No qualifying motor onset was detected within 500 ms after the change.",
    "CONTROL_QUIET": "No steady-note false trigger; control passed.",
    "CONTROL_FALSE_TRIGGER": "A false trigger occurred during a steady-note control.",
    "ANALYSIS_ERROR": "An unexpected analysis error occurred; see the log.",
}


def bandpass(samples, low_hz, high_hz):

    sections = butter(
        4, [low_hz, high_hz], btype="band", fs=SR, output="sos"
    )
    return sosfiltfilt(sections, samples)


def envelope(samples, low_hz, high_hz):

    return np.abs(hilbert(bandpass(samples, low_hz, high_hz)))


def first_sustained(mask, min_samples):

    if min_samples <= 1:
        positions = np.flatnonzero(mask)
        return int(positions[0]) if positions.size else None
    if mask.size < min_samples:
        return None

    counts = np.convolve(
        mask.astype(np.int32), np.ones(min_samples, dtype=np.int32), mode="valid"
    )
    positions = np.flatnonzero(counts == min_samples)
    return int(positions[0]) if positions.size else None


def sustained_runs(mask, min_samples):

    if not mask.size:
        return []

    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    starts, stops = edges[::2], edges[1::2]
    return [
        (int(start), int(stop))
        for start, stop in zip(starts, stops)
        if stop - start >= min_samples
    ]


def time_slice(samples, start_s, stop_s):

    start = max(0, int(round(start_s * SR)))
    stop = min(len(samples), int(round(stop_s * SR)))
    return samples[start:stop] if stop > start else samples[:0]


def snr_db(level, noise_floor):
    return 20.0 * np.log10(max(level, 1e-12) / max(noise_floor, 1e-12))


def harmonic_overlap(frequencies, max_harmonic=10, margin_hz=30.0):

    overlaps = []
    for frequency in sorted(set(frequencies)):
        for harmonic in range(2, max_harmonic + 1):
            hz = harmonic * frequency
            if MOTOR_LOW - margin_hz <= hz <= MOTOR_HIGH + margin_hz:
                overlaps.append(f"{harmonic}x{frequency:.1f}={hz:.0f}Hz")
    return overlaps


def tone_band(frequency):
    half_width = TONE_HALF_BW_LOW if frequency < LOW_FREQ_CUTOFF else TONE_HALF_BW
    return frequency - half_width, frequency + half_width


def make_stimulus(start_hz, end_hz, dur1=DUR1, dur2=DUR2):

    t1 = np.arange(int(SR * dur1)) / SR
    t2 = np.arange(int(SR * dur2)) / SR

    phase1 = 2 * np.pi * start_hz * t1

    phase2 = phase1[-1] + 2 * np.pi * start_hz / SR + 2 * np.pi * end_hz * t2
    signal = np.sin(np.concatenate((phase1, phase2)))

    fade_samples = int(FADE * SR)
    fade = np.ones_like(signal)
    fade[:fade_samples] = np.linspace(0, 1, fade_samples)
    fade[-fade_samples:] = np.linspace(1, 0, fade_samples)
    return signal * fade


def detect_tone(audio, frequency):
    low_hz, high_hz = tone_band(frequency)
    tone_env = envelope(audio, low_hz, high_hz)

    background = time_slice(tone_env, *TONE_NOISE_WIN)
    noise_mean = float(np.mean(background))
    noise_std = float(np.std(background))
    noise_k = TONE_K_LOW if frequency < LOW_FREQ_CUTOFF else TONE_K_HIGH


    level_start = TONE_VALID_WIN[0] + TONE_LEVEL_WIN[0]
    level_stop = TONE_VALID_WIN[0] + TONE_LEVEL_WIN[1]
    level = float(np.percentile(time_slice(tone_env, level_start, level_stop), 90))
    threshold = max(
        noise_mean + noise_k * noise_std,
        noise_mean + TONE_REL_FRAC * max(0.0, level - noise_mean),
    )

    start = int(TONE_SEARCH_WIN[0] * SR)
    stop = int(TONE_SEARCH_WIN[1] * SR)
    hit = first_sustained(
        tone_env[start:stop] >= threshold,
        int(TONE_MIN_SUSTAIN * SR),
    )

    return {
        "env": tone_env,
        "k": noise_k,
        "thresh": threshold,
        "snr_db": snr_db(level, noise_mean),
        "onset": None if hit is None else (start + hit) / SR,
    }


def scan_blips(motor_env, start_s, stop_s, threshold):

    segment = time_slice(motor_env, start_s, stop_s)
    runs = sustained_runs(segment > threshold, int(FALSE_TRIG_MIN * SR))
    offset = max(0, int(round(start_s * SR)))
    blips = [((offset + start) / SR, (offset + stop) / SR) for start, stop in runs]
    first = blips[0][0] if blips else None
    return len(blips), blips, first


def detect_motor_onset(motor_env, jump_s):


    jump_sample = int(round(jump_s * SR))
    baseline = time_slice(
        motor_env, jump_s + MOTOR_BASE_WIN[0], jump_s + MOTOR_BASE_WIN[1]
    )
    noise_mean = float(np.mean(baseline))
    noise_std = float(np.std(baseline))
    search = motor_env[jump_sample:jump_sample + int(MOTOR_SEARCH_LEN * SR)]

    level = float(np.percentile(search, MOTOR_LEVEL_PCT)) if search.size else noise_mean
    threshold = max(
        noise_mean + MOTOR_REL_FRAC * max(0.0, level - noise_mean),
        noise_mean + MOTOR_K * noise_std,
    )
    result = {
        "level": level,
        "thresh": threshold,
        "snr_db": snr_db(level, noise_mean),
        "vib": None,
    }

    if result["snr_db"] >= MOTOR_MIN_SNR_DB:
        hit = first_sustained(search > threshold, int(MOTOR_MIN_SUSTAIN * SR))
        if hit is not None:
            result["vib"] = (jump_sample + hit) / SR
    return result


def session_motor_gate(noise_floor, session_buzz_level):

    minimum = max(noise_floor, 1e-12) * 10 ** (MOTOR_MIN_SNR_DB / 20.0)
    if session_buzz_level is None or not np.isfinite(session_buzz_level):
        return minimum
    relative = noise_floor + MOTOR_REL_FRAC * max(0.0, session_buzz_level - noise_floor)
    return max(minimum, relative)


_CALIBRATION_CACHE = {}


def calibration_offsets(frequency):

    if frequency in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[frequency]

    rng = np.random.default_rng(12345)
    playback_latency = 0.15
    stimulus = make_stimulus(frequency, frequency)
    audio = np.concatenate((
        np.zeros(int((PRE_PAD + playback_latency) * SR)),
        0.05 * stimulus,
        np.zeros(int(POST_PAD * SR)),
    ))
    audio += 5e-4 * rng.standard_normal(len(audio))

    tone = detect_tone(audio, frequency)
    tone_offset = (
        tone["onset"] - PRE_PAD - playback_latency
        if tone["onset"] is not None else 0.0
    )

    jump_s = PRE_PAD + playback_latency + DUR1
    true_motor_start = jump_s + 0.050
    times = np.arange(len(audio)) / SR
    motor_signal = 1e-4 * rng.standard_normal(len(audio))
    active = (times >= true_motor_start) & (times < true_motor_start + 0.4)
    motor_signal[active] += 0.01 * np.sin(
        2 * np.pi * (MOTOR_LOW + MOTOR_HIGH) / 2 * times[active]
    )

    motor = detect_motor_onset(
        envelope(motor_signal, MOTOR_LOW, MOTOR_HIGH), jump_s
    )
    motor_offset = (
        motor["vib"] - true_motor_start if motor["vib"] is not None else 0.0
    )
    if tone["onset"] is None or motor["vib"] is None:
        print(f"Warning: calibration failed at {frequency} Hz; using zero for missing offsets")

    _CALIBRATION_CACHE[frequency] = (tone_offset, motor_offset)
    return tone_offset, motor_offset


def empty_result():
    numbers = (
        "tone_onset_ms", "tone_snr_db", "tone_thresh", "tone_k",
        "jump_ms", "vib_ms", "delay_raw_ms", "delay_ms", "motor_snr_db",
        "motor_thresh", "cal_tone_ms", "cal_motor_ms", "ft_thresh",
        "ft_first_ms", "pre_max_db", "buzz_level", "floor",
    )
    result = {name: np.nan for name in numbers}
    result.update({
        "status": "ANALYSIS_ERROR",
        "ft_count": "N/A",
        "ft_blips": [],
        "ft_scan": None,
        "off_count": "N/A",
        "off_blips": [],
        "off_scan": None,
        "gate_note": "",
    })
    return result


def analyze_recording(audio, frequency, control=False, session_level=None):


    result = empty_result()
    if np.max(np.abs(audio)) >= CLIP_LEVEL:
        result["status"] = "CLIPPED"
        return result, None, None

    tone = detect_tone(audio, frequency)
    result.update(
        tone_k=tone["k"], tone_thresh=tone["thresh"], tone_snr_db=tone["snr_db"]
    )
    if tone["onset"] is None:
        result["status"] = (
            "LOW_TONE_SNR" if tone["snr_db"] < TONE_MIN_SNR_DB else "NO_TONE"
        )
        return result, tone["env"], None

    tone_onset = tone["onset"]
    result["tone_onset_ms"] = tone_onset * 1000
    if tone["snr_db"] < TONE_MIN_SNR_DB:
        result["status"] = "LOW_TONE_SNR"
        return result, tone["env"], None
    if not (TONE_VALID_WIN[0] <= tone_onset <= TONE_VALID_WIN[1]):
        result["status"] = "TONE_OUT_OF_WINDOW"
        return result, tone["env"], None

    motor_env = envelope(audio, MOTOR_LOW, MOTOR_HIGH)
    noise_floor = float(np.mean(time_slice(motor_env, *TONE_NOISE_WIN)))
    gate = session_motor_gate(noise_floor, session_level)
    result.update(floor=noise_floor, ft_thresh=gate)
    if session_level is None:
        result["gate_note"] = "GATE_FALLBACK_FLOOR+12dB"

    steady_start = tone_onset + ONSET_IGNORE

    if control:
        note_end = tone_onset + DUR1 + DUR2
        steady_end = note_end - SCAN_END_GUARD
        count, blips, first = scan_blips(motor_env, steady_start, steady_end, gate)
        off_count, off_blips, _ = scan_blips(
            motor_env, steady_end, note_end + NOTE_OFF_WIN, gate
        )
        result.update(
            ft_count=count,
            ft_blips=blips,
            ft_scan=(steady_start, steady_end),
            ft_first_ms=first * 1000 if first is not None else np.nan,
            off_count=off_count,
            off_blips=off_blips,
            off_scan=(steady_end, note_end + NOTE_OFF_WIN),
            pre_max_db=snr_db(
                float(np.max(time_slice(motor_env, steady_start, steady_end))),
                noise_floor,
            ),

            status="CONTROL_FALSE_TRIGGER" if count else "CONTROL_QUIET",
        )
        return result, tone["env"], motor_env

    jump_s = tone_onset + DUR1
    result["jump_ms"] = jump_s * 1000
    steady_end = jump_s - SCAN_END_GUARD
    count, blips, first = scan_blips(motor_env, steady_start, steady_end, gate)
    result.update(
        ft_count=count,
        ft_blips=blips,
        ft_scan=(steady_start, steady_end),
        ft_first_ms=first * 1000 if first is not None else np.nan,
        pre_max_db=snr_db(
            float(np.max(time_slice(motor_env, steady_start, steady_end))),
            noise_floor,
        ),
    )

    motor = detect_motor_onset(motor_env, jump_s)
    result.update(
        motor_thresh=motor["thresh"],
        motor_snr_db=motor["snr_db"],
        buzz_level=motor["level"],
    )

    before_jump = time_slice(
        motor_env,
        jump_s + MOTOR_PRECHECK_WIN[0],
        jump_s + MOTOR_PRECHECK_WIN[1],
    )
    if first_sustained(
        before_jump > gate, int(MOTOR_PRE_SUSTAIN * SR)
    ) is not None:
        result["status"] = "ALREADY_BUZZING"
        return result, tone["env"], motor_env

    if motor["vib"] is None:
        result["status"] = "NO_MOTOR"
        return result, tone["env"], motor_env

    tone_offset, motor_offset = calibration_offsets(frequency)
    raw_delay_ms = (motor["vib"] - jump_s) * 1000
    result.update(
        vib_ms=motor["vib"] * 1000,
        delay_raw_ms=raw_delay_ms,
        cal_tone_ms=tone_offset * 1000,
        cal_motor_ms=motor_offset * 1000,
        delay_ms=raw_delay_ms - motor_offset * 1000 + tone_offset * 1000,
        status="OK",
    )
    return result, tone["env"], motor_env


def save_plot(folder, trial_name, result, tone_env=None, motor_env=None):

    try:
        figure, (tone_ax, motor_ax) = plt.subplots(
            2, 1, figsize=(11, 6), sharex=True
        )

        if tone_env is None:
            tone_ax.text(
                0.5, 0.5, "No tone data", ha="center", transform=tone_ax.transAxes
            )
        else:
            seconds = np.arange(len(tone_env)) / SR
            tone_ax.plot(seconds, tone_env, color="navy", lw=0.8, label="Tone envelope")
            if np.isfinite(result["tone_thresh"]):
                tone_ax.axhline(
                    result["tone_thresh"], color="red", ls="--", label="Tone threshold"
                )
            tone_ax.axvspan(*TONE_NOISE_WIN, color="grey", alpha=0.08)
            tone_ax.axvspan(*TONE_VALID_WIN, color="green", alpha=0.08)
            if np.isfinite(result["tone_onset_ms"]):
                tone_ax.axvline(
                    result["tone_onset_ms"] / 1000,
                    color="purple", ls=":", label="Tone onset",
                )
            tone_ax.set_yscale("log")
            tone_ax.legend(loc="upper right", fontsize=8)
        tone_ax.set_ylabel("Tone band (log)")

        if motor_env is None:
            motor_ax.text(
                0.5, 0.5, "Motor analysis not reached",
                ha="center", transform=motor_ax.transAxes,
            )
        else:
            seconds = np.arange(len(motor_env)) / SR
            motor_ax.plot(
                seconds, motor_env, color="navy", lw=0.8,
                alpha=0.8, label="Motor envelope",
            )
            if np.isfinite(result["motor_thresh"]):
                motor_ax.axhline(
                    result["motor_thresh"], color="red", ls="--",
                    label="Onset threshold",
                )
            if np.isfinite(result["ft_thresh"]):
                motor_ax.axhline(
                    result["ft_thresh"], color="grey", ls=":",
                    label="Session gate",
                )
            if result["ft_scan"]:
                motor_ax.axvspan(
                    *result["ft_scan"], color="gold", alpha=0.08,
                    label="Steady-note scan",
                )
            if result["off_scan"]:
                motor_ax.axvspan(
                    *result["off_scan"], color="violet", alpha=0.08,
                    label="Note-off scan",
                )
            for index, (start, stop) in enumerate(result["off_blips"]):
                motor_ax.axvspan(
                    start, stop, color="purple", alpha=0.35,
                    label="Note-off trigger" if index == 0 else None,
                )
            for index, (start, stop) in enumerate(result["ft_blips"]):
                motor_ax.axvspan(
                    start, stop, color="red", alpha=0.35,
                    label="False trigger" if index == 0 else None,
                )
            if np.isfinite(result["jump_ms"]):
                motor_ax.axvline(
                    result["jump_ms"] / 1000, color="green", label="Pitch change"
                )
            if np.isfinite(result["vib_ms"]):
                motor_ax.axvline(
                    result["vib_ms"] / 1000, color="orange", ls="-.",
                    label="Vibration start",
                )
            motor_ax.legend(loc="upper right", fontsize=8)
        motor_ax.set_xlabel("Time in recording (s)")
        motor_ax.set_ylabel("Motor band")

        latency = result["delay_ms"]
        latency_text = f"{latency:.1f} ms" if np.isfinite(latency) else "N/A"
        note_off_text = (
            f" | note-off: {result['off_count']}" if result["off_scan"] else ""
        )
        figure.suptitle(
            f"{trial_name} [{result['status']}] | delay: {latency_text}"
            f" | false triggers: {result['ft_count']}{note_off_text}"
            f" | tone SNR: {result['tone_snr_db']:.1f} dB"
        )
        figure.tight_layout()
        figure.savefig(os.path.join(folder, f"{trial_name}.png"), dpi=110)
    except Exception as error:
        print(f"  Could not save plot for {trial_name}: {error}")
    finally:
        plt.close("all")


def format_number(value, decimals=2):
    if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
        return f"{value:.{decimals}f}"
    return "NaN"


RESULT_HEADER = [
    "Run order", "Timestamp", "Trial ID", "Condition", "Category", "Rep",
    "Start freq (Hz)", "End freq (Hz)", "Jump (cents)", "Status",
    "Tone onset (ms)", "Tone SNR (dB)", "Tone thresh k",
    "Jump time in recording (ms)", "Vibration start (ms)", "Delay raw (ms)",
    "Cal tone (ms)", "Cal motor (ms)", "Delay (ms)", "Motor SNR (dB)",
    "Buzz level (p90)", "Session gate", "False triggers (steady pitch)",
    "First false trigger (ms)", "Max steady-pitch motor (dB over floor)",
    "Note-off triggers (controls)", "Target", "Met target?", "Analysis version",
    "Notes",
]


def write_summary(rows, path):
    headings = [
        "Condition", "Category", "Start freq (Hz)", "Jump (cents)",
        "Planned", "Valid", "Excluded (status:count)", "Mean delay (ms)",
        "Median (ms)", "SD (ms)", "Min (ms)", "Max (ms)",
        "% under target / controls quiet", "Trials with false triggers",
        "Total false triggers", "Note-off triggers (controls)",
    ]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headings)

        for label, category, frequency, cents, planned in CONDITIONS:
            trials = [row for row in rows if row["label"] == label]
            acceptable = (
                ("CONTROL_QUIET", "CONTROL_FALSE_TRIGGER")
                if category == "control" else ("OK",)
            )
            valid = [row for row in trials if row["status"] in acceptable]

            excluded = {}
            for row in trials:
                if row["status"] not in acceptable:
                    status = row["status"]
                    excluded[status] = excluded.get(status, 0) + 1

            delays = np.array([
                row["delay_ms"] for row in valid
                if np.isfinite(row["delay_ms"])
            ])
            false_counts = [
                row["ft_count"] for row in trials if isinstance(row["ft_count"], int)
            ]
            off_counts = [
                row["off_count"] for row in trials if isinstance(row["off_count"], int)
            ]

            if category == "pitch" and delays.size:
                target_rate = f"{100 * np.mean(delays <= TARGET_MS):.0f}%"
            elif category == "control" and valid:
                quiet = sum(row["status"] == "CONTROL_QUIET" for row in valid)
                target_rate = f"{quiet}/{len(valid)} quiet"
            else:
                target_rate = "N/A"

            if delays.size:
                stats = [
                    format_number(delays.mean()),
                    format_number(np.median(delays)),
                    format_number(delays.std(ddof=1)) if delays.size > 1 else "NaN",
                    format_number(delays.min()),
                    format_number(delays.max()),
                ]
            else:
                stats = ["NaN"] * 5

            writer.writerow([
                label, category, format_number(frequency), format_number(cents),
                len(trials) or planned, len(valid),
                "; ".join(f"{status}:{count}" for status, count in excluded.items())
                or "none",
                *stats,
                target_rate,
                sum(count > 0 for count in false_counts),
                sum(false_counts),
                sum(off_counts) if category == "control" else "N/A",
            ])


def analyze_session(manifest, wav_dir, output_tag, log):

    plot_dir = f"trial_plots_{output_tag}"
    os.makedirs(plot_dir, exist_ok=True)

    def load_recording(trial):
        path = os.path.join(wav_dir, f"{trial['name']}.wav")
        if not os.path.exists(path):
            return None
        audio, sample_rate = sf.read(path, dtype="float64")
        if sample_rate != SR:
            raise RuntimeError(f"{path}: sample rate {sample_rate} != {SR}")
        return audio if audio.ndim == 1 else audio[:, 0]


    levels = []
    for trial in manifest:
        if trial["cat"] == "control" or trial.get("rec_status") == "REC_ERROR":
            continue
        try:
            audio = load_recording(trial)
            if audio is None:
                continue
            result, _, _ = analyze_recording(audio, trial["f1"])
            if (
                result["status"] in ("OK", "ALREADY_BUZZING")
                and np.isfinite(result["buzz_level"])
                and result["motor_snr_db"] >= MOTOR_MIN_SNR_DB
            ):
                levels.append(result["buzz_level"])
        except Exception:
            log.write(f"[pass 1: {trial['name']}]\n{traceback.format_exc()}\n")

    session_level = float(np.median(levels)) if levels else None
    level_text = (
        f"{session_level:.3e}" if session_level is not None
        else "unavailable; using floor + 12 dB"
    )
    print(f"Session buzz level (from {len(levels)} trials): {level_text}")


    rows = []
    status_counts = {}
    csv_path = f"results_pitch_change_{output_tag}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(RESULT_HEADER)

        for trial in manifest:
            name = trial["name"]
            category = trial["cat"]
            is_control = category == "control"
            notes = list(trial.get("notes", []))
            overlaps = harmonic_overlap((trial["f1"], trial["f2"]))
            if overlaps:
                notes.append("HARMONIC_NEAR_MOTOR_BAND:" + ";".join(overlaps))

            result = empty_result()
            tone_env = motor_env = None
            if trial.get("rec_status") == "REC_ERROR":
                result["status"] = "REC_ERROR"
            else:
                try:
                    audio = load_recording(trial)
                    if audio is None:
                        result["status"] = "MISSING_WAV"
                    else:
                        result, tone_env, motor_env = analyze_recording(
                            audio, trial["f1"], control=is_control,
                            session_level=session_level,
                        )
                except Exception as error:
                    result = empty_result()
                    result["status"] = "ANALYSIS_ERROR"
                    notes.append(f"{type(error).__name__}: {error}")
                    log.write(f"[{name}] ANALYSIS_ERROR\n{traceback.format_exc()}\n")

            if result["gate_note"]:
                notes.append(result["gate_note"])
            save_plot(plot_dir, name, result, tone_env, motor_env)

            status = result["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            if is_control:
                target = "no buzz"
                met_target = {
                    "CONTROL_QUIET": "Y", "CONTROL_FALSE_TRIGGER": "N"
                }.get(status, "N/A")
            elif status == "OK":
                target = "N/A" if category == "pitch_report" else f"{TARGET_MS:.0f} ms"
                met_target = (
                    "Report only" if category == "pitch_report"
                    else "Y" if result["delay_ms"] <= TARGET_MS else "N"
                )
            else:
                target = met_target = "N/A"

            buzz_text = (
                f"{result['buzz_level']:.3e}"
                if np.isfinite(result["buzz_level"]) else "NaN"
            )
            gate_text = (
                f"{result['ft_thresh']:.3e}"
                if np.isfinite(result["ft_thresh"]) else "NaN"
            )
            writer.writerow([
                trial.get("order", ""), trial.get("timestamp", ""), name,
                trial["label"], category, trial["rep"],
                format_number(trial["f1"]), format_number(trial["f2"]),
                format_number(trial["cents"]), status,
                format_number(result["tone_onset_ms"]),
                format_number(result["tone_snr_db"], 1),
                format_number(result["tone_k"], 1),
                format_number(result["jump_ms"]),
                format_number(result["vib_ms"]),
                format_number(result["delay_raw_ms"]),
                format_number(result["cal_tone_ms"]),
                format_number(result["cal_motor_ms"]),
                format_number(result["delay_ms"]),
                format_number(result["motor_snr_db"], 1),
                buzz_text, gate_text, result["ft_count"],
                format_number(result["ft_first_ms"]),
                format_number(result["pre_max_db"], 1),
                result["off_count"] if is_control else "N/A",
                target, met_target, ANALYSIS_VERSION, " | ".join(notes),
            ])
            handle.flush()

            rows.append({
                "label": trial["label"],
                "status": status,
                "delay_ms": result["delay_ms"],
                "ft_count": result["ft_count"],
                "off_count": result["off_count"],
            })
            log.write(
                f"[{name}] {status} delay={format_number(result['delay_ms'])} "
                f"false_triggers={result['ft_count']} "
                f"note_off={result['off_count']} notes={notes}\n"
            )

    summary_path = f"summary_{output_tag}.csv"
    write_summary(rows, summary_path)
    print("Trial statuses:", status_counts)
    print(f"Saved {csv_path}, {summary_path} and plots in {plot_dir}/")
    return session_level, len(levels), status_counts


def condition_lookup():
    return {
        label: (category, frequency, cents)
        for label, category, frequency, cents, _ in CONDITIONS
    }


def build_trials():

    trials = []
    for label, category, start_hz, cents, repetitions in CONDITIONS:
        end_hz = start_hz * 2 ** (cents / 1200.0)
        for repetition in range(1, repetitions + 1):
            trials.append({
                "label": label,
                "cat": category,
                "rep": repetition,
                "f1": start_hz,
                "f2": end_hz,
                "cents": cents,
                "name": f"Trial_{label}_{repetition:02d}",
            })

    rng = np.random.default_rng(SHUFFLE_SEED)
    trials = [trials[index] for index in rng.permutation(len(trials))]
    for index, trial in enumerate(trials, start=1):
        trial["order"] = index
    return trials


def manifest_from_wavs(wav_dir, original_csv=None):

    conditions = condition_lookup()
    previous = {}

    if original_csv and os.path.exists(original_csv):
        with open(original_csv, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                previous[row["Trial ID"]] = (
                    row.get("Run order", ""),
                    row.get("Timestamp", ""),
                    row.get("Status", ""),
                )

    def trial_from_name(name, recording_status=None):

        if not name.startswith("Trial_"):
            return None
        try:
            label, rep_text = name[len("Trial_"):].rsplit("_", 1)
            repetition = int(rep_text)
        except (ValueError, IndexError):
            print(f"  Skipping malformed trial name: {name}")
            return None
        if label not in conditions:
            print(f"  Skipping {name}: condition '{label}' is not configured")
            return None

        category, start_hz, cents = conditions[label]
        run_order, timestamp, _ = previous.get(name, ("", "", ""))
        entry = {
            "name": name,
            "label": label,
            "cat": category,
            "rep": repetition,
            "f1": start_hz,
            "f2": start_hz * 2 ** (cents / 1200.0),
            "cents": cents,
            "order": int(run_order) if str(run_order).isdigit() else "",
            "timestamp": timestamp,
        }
        if recording_status:
            entry["rec_status"] = recording_status
        return entry

    manifest = []
    for filename in os.listdir(wav_dir):
        if filename.startswith("Trial_") and filename.endswith(".wav"):
            trial = trial_from_name(filename[:-4])
            if trial:
                manifest.append(trial)


    present = {trial["name"] for trial in manifest}
    for trial_name, (_, _, old_status) in previous.items():
        if trial_name not in present and old_status == "REC_ERROR":
            trial = trial_from_name(trial_name, recording_status="REC_ERROR")
            if trial:
                manifest.append(trial)

    manifest.sort(
        key=lambda trial: (
            trial["order"] if trial["order"] != "" else 10**6,
            trial["name"],
        )
    )
    return manifest


def warm_up_audio():

    print(f"Warming up the audio stream ({WARMUP_CALLS} silent calls)...")
    silence = np.zeros(int(WARMUP_LEN * SR), dtype=np.float32)
    successes = 0

    for index in range(WARMUP_CALLS):
        try:
            recording = sd.playrec(
                silence, samplerate=SR, channels=1, device=DEVICE_CONFIG
            )
            sd.wait()
            if recording is not None and len(recording) == len(silence):
                successes += 1
        except Exception as error:
            print(f"  Warm-up call {index + 1} failed: {error}")
        time.sleep(0.5)

    time.sleep(1.0)
    print(f"Warm-up complete: {successes}/{WARMUP_CALLS} full buffers")
    return successes


def record_session(trials, wav_dir, log):

    os.makedirs(wav_dir, exist_ok=True)
    for trial in trials:
        print(
            f"[{trial['order']}/{len(trials)}] Recording {trial['name']} "
            f"({trial['f1']:.2f} -> {trial['f2']:.2f} Hz)"
        )
        time.sleep(INTER_TRIAL_GAP)
        trial["timestamp"] = datetime.now().isoformat(timespec="seconds")
        trial["notes"] = []

        try:
            stimulus = make_stimulus(trial["f1"], trial["f2"])
            padded = np.concatenate((
                np.zeros(int(PRE_PAD * SR)),
                stimulus,
                np.zeros(int(POST_PAD * SR)),
            )).astype(np.float32)

            recording = sd.playrec(
                padded, samplerate=SR, channels=1, device=DEVICE_CONFIG
            )
            sd.wait()
            audio = np.asarray(recording, dtype=np.float64).flatten()
            if len(audio) != len(padded) or not np.all(np.isfinite(audio)):
                raise RuntimeError(
                    f"Incomplete recording: {len(audio)} samples, "
                    f"expected {len(padded)}"
                )

            sf.write(
                os.path.join(wav_dir, f"{trial['name']}.wav"),
                audio, SR, subtype="FLOAT",
            )
            trial["rec_status"] = "RECORDED"
        except Exception as error:
            trial["rec_status"] = "REC_ERROR"
            trial["notes"].append(f"{type(error).__name__}: {error}")
            log.write(f"[{trial['name']}] REC_ERROR\n{traceback.format_exc()}\n")
            print(f"  Recording failed: {error}")
    return trials


def session_metadata(output_tag, mode):
    return {
        "run_tag": output_tag,
        "mode": mode,
        "analysis_version": ANALYSIS_VERSION,
        "started": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "params": {
            key: value for key, value in globals().items()
            if key.isupper() and isinstance(value, (int, float, tuple, str, list))
        },
        "status_codes": STATUS_DESCRIPTIONS,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure pitch-change to vibration latency"
    )
    parser.add_argument(
        "--reanalyze", metavar="WAV_DIR",
        help="Score saved WAV recordings without making a new recording",
    )
    parser.add_argument(
        "--orig-csv", metavar="CSV",
        help="Old results CSV, used to recover run order and timestamps",
    )
    args = parser.parse_args()

    if args.reanalyze:
        source_name = os.path.basename(os.path.normpath(args.reanalyze))
        output_tag = f"{RUN_TAG}_reanalysis_of_{source_name}"
        metadata = session_metadata(output_tag, "reanalyze")
        metadata.update({
            "source_wavs": os.path.abspath(args.reanalyze),
            "source_csv": args.orig_csv,
        })
        manifest = manifest_from_wavs(args.reanalyze, args.orig_csv)
        wav_dir = args.reanalyze
        print(f"Reanalyzing {len(manifest)} trials from {wav_dir}")
    else:
        output_tag = RUN_TAG
        print(sd.query_devices())
        print(f"Audio devices (mic, speaker): {DEVICE_CONFIG}")
        metadata = session_metadata(output_tag, "record")
        metadata.update({
            "sounddevice": sd.__version__,
            "devices": {
                "mic": str(sd.query_devices(MIC_DEV)["name"]),
                "spk": str(sd.query_devices(SPK_DEV)["name"]),
            },
        })
        manifest = build_trials()
        metadata["trial_order"] = [trial["name"] for trial in manifest]
        metadata["warmup_ok_calls"] = warm_up_audio()
        wav_dir = f"trial_wavs_{RUN_TAG}"

    log_path = f"run_log_{output_tag}.txt"
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            if not args.reanalyze:
                print(f"Recording {len(manifest)} trials (seed {SHUFFLE_SEED})")
                record_session(manifest, wav_dir, log)
            print("Analyzing recordings...")
            level, level_count, statuses = analyze_session(
                manifest, wav_dir, output_tag, log
            )
            metadata.update({
                "session_buzz_level": level,
                "session_level_n_trials": level_count,
                "status_counts": statuses,
                "calibration_ms": {
                    str(frequency): [round(offset * 1000, 3) for offset in offsets]
                    for frequency, offsets in _CALIBRATION_CACHE.items()
                },
                "finished": datetime.now().isoformat(timespec="seconds"),
            })
        finally:
            with open(f"run_meta_{output_tag}.json", "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, default=str)

    print("Finished.")


if __name__ == "__main__":
    main()
