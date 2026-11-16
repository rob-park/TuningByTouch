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


RUN_TAG = "nn_run1"
ANALYSIS_VERSION = "nn1"
FIRMWARE_VERSION = "settle=2"

MIC_DEV = 1
SPK_DEV = 3

CONDITIONS = [
    ("G3_sharp20", "new_note", "G3", 196.00, 20.0, 8),
    ("A4_sharp20", "new_note", "A4", 440.00, 20.0, 8),
    ("A6_sharp20", "new_note", "A6", 1760.00, 20.0, 8),
    ("G3_intune", "new_note_control", "G3", 196.00, 0.0, 4),
    ("A4_intune", "new_note_control", "A4", 440.00, 0.0, 4),
    ("A6_intune", "new_note_control", "A6", 1760.00, 0.0, 4),
]

SHUFFLE_SEED = 20261116
INTER_TRIAL_GAP = 2.0

SR = 48000
DEVICE_CONFIG = (MIC_DEV, SPK_DEV)

PRE_PAD = 0.5
NOTE_DUR = 2.0
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
MOTOR_BASE_WIN = (-0.45, -0.05)
MOTOR_SEARCH_LEN = 0.80
MOTOR_LEVEL_PCT = 90
MOTOR_MIN_SNR_DB = 12.0
MOTOR_PRE_SUSTAIN = 0.020

FALSE_TRIG_MIN = 0.010
SCAN_END_GUARD = 0.05
ONSET_IGNORE = 0.010
NOTE_OFF_WIN = 0.25

CLIP_LEVEL = 0.99
TARGET_MS = 200.0


STATUS_DESCRIPTIONS = {
    "OK": "Note onset and first vibration detected; delay is valid.",
    "REC_ERROR": "The audio device failed to return a complete recording.",
    "MISSING_WAV": "The expected recording was not found during reanalysis.",
    "CLIPPED": "The microphone signal clipped; timing may be unreliable.",
    "NO_TONE": "No sustained tone onset was found in the search window.",
    "LOW_TONE_SNR": "Tone signal-to-noise ratio was below 10 dB.",
    "TONE_OUT_OF_WINDOW": "Tone onset was outside the expected 0.6-1.2 s window.",
    "ALREADY_BUZZING": "The motor was above the session gate in the silence before the note.",
    "NO_MOTOR": "No qualifying vibration within 800 ms after the note started.",
    "CONTROL_QUIET": "In-tune new note: no vibration during the note. PASS.",
    "CONTROL_FALSE_TRIGGER": "In-tune new note: vibration during the note. FAIL.",
    "ANALYSIS_ERROR": "An unexpected analysis error occurred; see the log.",
}


def bandpass(samples, low_hz, high_hz):
    sos = butter(4, [low_hz, high_hz], btype="band", fs=SR, output="sos")
    return sosfiltfilt(sos, samples)


def envelope(samples, low_hz, high_hz):
    filtered = bandpass(samples, low_hz, high_hz)
    return np.abs(hilbert(filtered))


def first_sustained(mask, min_samples):
    if min_samples <= 1:
        hits = np.flatnonzero(mask)
        return int(hits[0]) if hits.size else None

    if mask.size < min_samples:
        return None

    window = np.ones(min_samples, dtype=np.int32)
    counts = np.convolve(mask.astype(np.int32), window, mode="valid")
    hits = np.flatnonzero(counts == min_samples)

    return int(hits[0]) if hits.size else None


def sustained_runs(mask, min_samples):
    if not mask.size:
        return []

    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))

    runs = []
    for start, stop in zip(edges[::2], edges[1::2]):
        if stop - start >= min_samples:
            runs.append((int(start), int(stop)))

    return runs


def time_slice(samples, start_s, stop_s):
    a = max(0, int(round(start_s * SR)))
    b = min(len(samples), int(round(stop_s * SR)))

    if b <= a:
        return samples[:0]

    return samples[a:b]


def snr_db(level, noise_floor):
    signal = max(level, 1e-12)
    noise = max(noise_floor, 1e-12)
    return 20.0 * np.log10(signal / noise)


def harmonic_overlap(frequency, max_harmonic=10, margin_hz=30.0):
    overlaps = []

    for n in range(2, max_harmonic + 1):
        hz = n * frequency
        if MOTOR_LOW - margin_hz <= hz <= MOTOR_HIGH + margin_hz:
            overlaps.append(f"{n}x{frequency:.1f}={hz:.0f}Hz")

    return overlaps


def tone_band(frequency):
    half_bw = TONE_HALF_BW_LOW if frequency < LOW_FREQ_CUTOFF else TONE_HALF_BW
    return frequency - half_bw, frequency + half_bw


def make_stimulus(frequency, duration=NOTE_DUR):
    t = np.arange(int(SR * duration)) / SR
    tone = np.sin(2 * np.pi * frequency * t)

    n_fade = int(FADE * SR)
    fade = np.ones_like(tone)
    fade[:n_fade] = np.linspace(0, 1, n_fade)
    fade[-n_fade:] = np.linspace(1, 0, n_fade)

    return tone * fade


def scan_blips(motor_env, start_s, stop_s, threshold):
    segment = time_slice(motor_env, start_s, stop_s)
    runs = sustained_runs(segment > threshold, int(FALSE_TRIG_MIN * SR))

    offset = max(0, int(round(start_s * SR)))
    blips = []

    for a, b in runs:
        blips.append(((offset + a) / SR, (offset + b) / SR))

    first = blips[0][0] if blips else None
    return len(blips), blips, first


def detect_tone(audio, frequency):
    low, high = tone_band(frequency)
    tone_env = envelope(audio, low, high)

    background = time_slice(tone_env, *TONE_NOISE_WIN)
    noise_mean = float(np.mean(background))
    noise_std = float(np.std(background))

    k = TONE_K_LOW if frequency < LOW_FREQ_CUTOFF else TONE_K_HIGH

    level_start = TONE_VALID_WIN[0] + TONE_LEVEL_WIN[0]
    level_end = TONE_VALID_WIN[0] + TONE_LEVEL_WIN[1]

    level = float(np.percentile(
        time_slice(tone_env, level_start, level_end), 90
    ))

    threshold = max(
        noise_mean + k * noise_std,
        noise_mean + TONE_REL_FRAC * max(0.0, level - noise_mean)
    )

    start = int(TONE_SEARCH_WIN[0] * SR)
    stop = int(TONE_SEARCH_WIN[1] * SR)

    hit = first_sustained(
        tone_env[start:stop] >= threshold,
        int(TONE_MIN_SUSTAIN * SR)
    )

    onset = (start + hit) / SR if hit is not None else None

    return {
        "env": tone_env,
        "thresh": threshold,
        "snr_db": snr_db(level, noise_mean),
        "onset": onset,
    }


def detect_motor_onset(motor_env, onset_s):
    onset_idx = int(round(onset_s * SR))

    baseline = time_slice(
        motor_env,
        onset_s + MOTOR_BASE_WIN[0],
        onset_s + MOTOR_BASE_WIN[1]
    )

    noise_mean = float(np.mean(baseline))
    noise_std = float(np.std(baseline))

    search_end = onset_idx + int(MOTOR_SEARCH_LEN * SR)
    search = motor_env[onset_idx:search_end]

    level = float(np.percentile(search, MOTOR_LEVEL_PCT)) if search.size else noise_mean

    threshold = max(
        noise_mean + MOTOR_REL_FRAC * max(0.0, level - noise_mean),
        noise_mean + MOTOR_K * noise_std
    )

    result = {
        "level": level,
        "thresh": threshold,
        "snr_db": snr_db(level, noise_mean),
        "vib": None,
    }

    if result["snr_db"] < MOTOR_MIN_SNR_DB:
        return result

    hit = first_sustained(
        search > threshold,
        int(MOTOR_MIN_SUSTAIN * SR)
    )

    if hit is not None:
        result["vib"] = (onset_idx + hit) / SR

    return result


def session_motor_gate(noise_floor, session_buzz_level):
    minimum = max(noise_floor, 1e-12) * 10 ** (MOTOR_MIN_SNR_DB / 20.0)

    if session_buzz_level is None or not np.isfinite(session_buzz_level):
        return minimum

    session_gate = noise_floor + MOTOR_REL_FRAC * max(
        0.0, session_buzz_level - noise_floor
    )

    return max(minimum, session_gate)


_CALIBRATION_CACHE = {}


def calibration_offsets(frequency):
    if frequency in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[frequency]

    rng = np.random.default_rng(12345)
    latency = 0.15

    audio = np.concatenate((
        np.zeros(int((PRE_PAD + latency) * SR)),
        0.05 * make_stimulus(frequency),
        np.zeros(int(POST_PAD * SR))
    ))

    audio += 5e-4 * rng.standard_normal(len(audio))

    tone = detect_tone(audio, frequency)
    tone_offset = (
        tone["onset"] - PRE_PAD - latency
        if tone["onset"] is not None else 0.0
    )

    onset_s = PRE_PAD + latency
    true_start = onset_s + 0.100

    times = np.arange(len(audio)) / SR
    motor = 1e-4 * rng.standard_normal(len(audio))

    active = (times >= true_start) & (times < true_start + 0.6)
    motor_freq = (MOTOR_LOW + MOTOR_HIGH) / 2

    motor[active] += 0.01 * np.sin(
        2 * np.pi * motor_freq * times[active]
    )

    motor_env = envelope(motor, MOTOR_LOW, MOTOR_HIGH)
    found = detect_motor_onset(motor_env, onset_s)

    motor_offset = (
        found["vib"] - true_start
        if found["vib"] is not None else 0.0
    )

    if tone["onset"] is None or found["vib"] is None:
        print(
            f"Warning: calibration failed at {frequency:.2f} Hz; "
            "using zero for missing offsets"
        )

    offsets = (tone_offset, motor_offset)
    _CALIBRATION_CACHE[frequency] = offsets

    return offsets


def empty_result():
    fields = (
        "tone_onset_ms", "tone_snr_db", "vib_ms", "delay_raw_ms",
        "delay_ms", "motor_snr_db", "motor_thresh", "cal_tone_ms",
        "cal_motor_ms", "gate", "buzz_level", "ft_first_ms"
    )

    result = {field: np.nan for field in fields}

    result.update(
        status="ANALYSIS_ERROR",
        ft_count="N/A",
        ft_blips=[],
        ft_scan=None,
        off_count="N/A",
        off_blips=[],
        off_scan=None,
        gate_note=""
    )

    return result


def analyze_recording(audio, played_hz, control=False, session_level=None):
    result = empty_result()

    if np.max(np.abs(audio)) >= CLIP_LEVEL:
        result["status"] = "CLIPPED"
        return result, None, None

    tone = detect_tone(audio, played_hz)
    result["tone_snr_db"] = tone["snr_db"]

    if tone["onset"] is None:
        result["status"] = (
            "LOW_TONE_SNR"
            if tone["snr_db"] < TONE_MIN_SNR_DB else "NO_TONE"
        )
        return result, tone["env"], None

    onset = tone["onset"]
    result["tone_onset_ms"] = onset * 1000

    if tone["snr_db"] < TONE_MIN_SNR_DB:
        result["status"] = "LOW_TONE_SNR"
        return result, tone["env"], None

    if not TONE_VALID_WIN[0] <= onset <= TONE_VALID_WIN[1]:
        result["status"] = "TONE_OUT_OF_WINDOW"
        return result, tone["env"], None

    motor_env = envelope(audio, MOTOR_LOW, MOTOR_HIGH)
    noise_floor = float(np.mean(time_slice(motor_env, *TONE_NOISE_WIN)))

    gate = session_motor_gate(noise_floor, session_level)
    result["gate"] = gate

    if session_level is None:
        result["gate_note"] = "GATE_FALLBACK_FLOOR+12dB"

    before = time_slice(
        motor_env,
        onset + MOTOR_BASE_WIN[0],
        onset + MOTOR_BASE_WIN[1]
    )

    pre_hit = first_sustained(
        before > gate,
        int(MOTOR_PRE_SUSTAIN * SR)
    )

    if pre_hit is not None:
        result["status"] = "ALREADY_BUZZING"
        return result, tone["env"], motor_env

    note_end = onset + NOTE_DUR

    if control:
        steady_end = note_end - SCAN_END_GUARD

        count, blips, first = scan_blips(
            motor_env,
            onset + ONSET_IGNORE,
            steady_end,
            gate
        )

        off_count, off_blips, _ = scan_blips(
            motor_env,
            steady_end,
            note_end + NOTE_OFF_WIN,
            gate
        )

        result.update(
            ft_count=count,
            ft_blips=blips,
            ft_scan=(onset + ONSET_IGNORE, steady_end),
            ft_first_ms=first * 1000 if first is not None else np.nan,
            off_count=off_count,
            off_blips=off_blips,
            off_scan=(steady_end, note_end + NOTE_OFF_WIN),
            status="CONTROL_FALSE_TRIGGER" if count else "CONTROL_QUIET"
        )

        return result, tone["env"], motor_env

    motor = detect_motor_onset(motor_env, onset)

    result.update(
        motor_thresh=motor["thresh"],
        motor_snr_db=motor["snr_db"],
        buzz_level=motor["level"]
    )

    if motor["vib"] is None:
        result["status"] = "NO_MOTOR"
        return result, tone["env"], motor_env

    tone_offset, motor_offset = calibration_offsets(played_hz)

    raw_ms = (motor["vib"] - onset) * 1000
    corrected_ms = raw_ms - motor_offset * 1000 + tone_offset * 1000

    result.update(
        vib_ms=motor["vib"] * 1000,
        delay_raw_ms=raw_ms,
        cal_tone_ms=tone_offset * 1000,
        cal_motor_ms=motor_offset * 1000,
        delay_ms=corrected_ms,
        status="OK"
    )

    return result, tone["env"], motor_env


def save_plot(folder, name, result, tone_env=None, motor_env=None):
    try:
        fig, (ax_tone, ax_motor) = plt.subplots(
            2, 1, figsize=(11, 6), sharex=True
        )

        if tone_env is None:
            ax_tone.text(
                0.5, 0.5, "No tone data",
                ha="center", transform=ax_tone.transAxes
            )
        else:
            t = np.arange(len(tone_env)) / SR

            ax_tone.plot(
                t, tone_env, color="navy", lw=0.8,
                label="Tone envelope"
            )

            ax_tone.axvspan(*TONE_NOISE_WIN, color="grey", alpha=0.08)
            ax_tone.axvspan(*TONE_VALID_WIN, color="green", alpha=0.08)

            if np.isfinite(result["tone_onset_ms"]):
                ax_tone.axvline(
                    result["tone_onset_ms"] / 1000,
                    color="purple", ls=":", label="Note onset"
                )

            ax_tone.set_yscale("log")
            ax_tone.legend(loc="upper right", fontsize=8)

        ax_tone.set_ylabel("Tone band (log)")

        if motor_env is None:
            ax_motor.text(
                0.5, 0.5, "Motor analysis not reached",
                ha="center", transform=ax_motor.transAxes
            )
        else:
            t = np.arange(len(motor_env)) / SR

            ax_motor.plot(
                t, motor_env, color="navy", lw=0.8,
                alpha=0.8, label="Motor envelope"
            )

            if np.isfinite(result["motor_thresh"]):
                ax_motor.axhline(
                    result["motor_thresh"], color="red",
                    ls="--", label="Onset threshold"
                )

            if np.isfinite(result["gate"]):
                ax_motor.axhline(
                    result["gate"], color="grey",
                    ls=":", label="Session gate"
                )

            if np.isfinite(result["tone_onset_ms"]):
                ax_motor.axvline(
                    result["tone_onset_ms"] / 1000,
                    color="purple", ls=":", label="Note onset"
                )

            if result["ft_scan"]:
                ax_motor.axvspan(
                    *result["ft_scan"], color="gold",
                    alpha=0.08, label="In-tune note scan"
                )

            for i, (a, b) in enumerate(result["ft_blips"]):
                label = "False trigger" if i == 0 else None
                ax_motor.axvspan(
                    a, b, color="red", alpha=0.35, label=label
                )

            for i, (a, b) in enumerate(result["off_blips"]):
                label = "Note-off trigger" if i == 0 else None
                ax_motor.axvspan(
                    a, b, color="purple", alpha=0.35, label=label
                )

            if np.isfinite(result["vib_ms"]):
                ax_motor.axvline(
                    result["vib_ms"] / 1000,
                    color="orange", ls="-.", label="Vibration start"
                )

            ax_motor.legend(loc="upper right", fontsize=8)

        ax_motor.set_xlabel("Time in recording (s)")
        ax_motor.set_ylabel("Motor band")

        raw = result["delay_raw_ms"]
        delay_text = f"{raw:.1f} ms" if np.isfinite(raw) else "N/A"

        fig.suptitle(
            f"{name} [{result['status']}] | raw delay: {delay_text} | "
            f"tone SNR: {result['tone_snr_db']:.1f} dB"
        )

        fig.tight_layout()
        fig.savefig(os.path.join(folder, f"{name}.png"), dpi=110)

    except Exception as exc:
        print(f"  Could not save plot for {name}: {exc}")

    finally:
        plt.close("all")


def fmt(value, decimals=2):
    if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
        return f"{value:.{decimals}f}"
    return "NaN"


RESULT_HEADER = [
    "Run order", "Timestamp", "Trial ID", "Condition", "Category",
    "Note", "Rep", "In-tune freq (Hz)", "Offset (cents)",
    "Played freq (Hz)", "Status", "Note start (ms)",
    "Vibration start (ms)", "Delay raw (ms)", "Cal tone (ms)",
    "Cal motor (ms)", "Delay corrected (ms)", "Tone SNR (dB)",
    "Motor SNR (dB)", "Session gate", "False triggers (controls)",
    "First false trigger (ms)", "Note-off triggers (controls)",
    "Target (ms)", "Met target?", "Analysis version", "Firmware", "Notes"
]


def build_trials():
    trials = []

    for label, category, note, base_hz, cents, reps in CONDITIONS:
        played_hz = base_hz * 2 ** (cents / 1200.0)

        for rep in range(1, reps + 1):
            trials.append({
                "label": label,
                "cat": category,
                "note": note,
                "rep": rep,
                "base": base_hz,
                "cents": cents,
                "played": played_hz,
                "name": f"Trial_{label}_{rep:02d}"
            })

    rng = np.random.default_rng(SHUFFLE_SEED)
    order = rng.permutation(len(trials))
    trials = [trials[i] for i in order]

    for i, trial in enumerate(trials, 1):
        trial["order"] = i

    return trials


def write_summary(rows, path):
    headers = [
        "Condition", "Category", "Note", "Offset (cents)",
        "Planned", "Valid", "Excluded (status:count)",
        "Median raw delay (ms)", "Mean raw (ms)",
        "SD raw (ms)", "Min raw (ms)", "Max raw (ms)",
        "Median corrected (ms)",
        "% under target (raw) / controls quiet",
        "Note-off triggers (controls)"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for label, category, note, _, cents, planned in CONDITIONS:
            group = [r for r in rows if r["label"] == label]
            control = category == "new_note_control"

            valid_statuses = (
                ("CONTROL_QUIET", "CONTROL_FALSE_TRIGGER")
                if control else ("OK",)
            )

            valid = [r for r in group if r["status"] in valid_statuses]

            excluded = {}
            for row in group:
                status = row["status"]
                if status not in valid_statuses:
                    excluded[status] = excluded.get(status, 0) + 1

            raw = np.array([
                r["raw"] for r in valid if np.isfinite(r["raw"])
            ])

            corrected = np.array([
                r["corr"] for r in valid if np.isfinite(r["corr"])
            ])

            if not control and raw.size:
                rate = f"{100 * np.mean(raw <= TARGET_MS):.0f}%"
            elif control and valid:
                quiet = sum(r["status"] == "CONTROL_QUIET" for r in valid)
                rate = f"{quiet}/{len(valid)} quiet"
            else:
                rate = "N/A"

            if raw.size:
                stats = [
                    fmt(np.median(raw)),
                    fmt(raw.mean()),
                    fmt(raw.std(ddof=1)) if raw.size > 1 else "NaN",
                    fmt(raw.min()),
                    fmt(raw.max())
                ]
            else:
                stats = ["NaN"] * 5

            off_counts = [
                r["off"] for r in group
                if isinstance(r["off"], int)
            ]

            excluded_text = "; ".join(
                f"{status}:{count}" for status, count in excluded.items()
            ) or "none"

            median_corr = (
                fmt(np.median(corrected)) if corrected.size else "NaN"
            )

            writer.writerow([
                label,
                category,
                note,
                fmt(cents),
                len(group) or planned,
                len(valid),
                excluded_text,
                *stats,
                median_corr,
                rate,
                sum(off_counts) if control else "N/A"
            ])


def analyze_session(manifest, wav_dir, output_tag, log):
    plot_dir = f"trial_plots_{output_tag}"
    os.makedirs(plot_dir, exist_ok=True)

    def load(trial):
        filename = f"{trial['name']}.wav"
        path = os.path.join(wav_dir, filename)

        if not os.path.exists(path):
            return None

        audio, rate = sf.read(path, dtype="float64")

        if rate != SR:
            raise RuntimeError(f"{path}: sample rate {rate} != {SR}")

        if audio.ndim == 1:
            return audio

        return audio[:, 0]

    levels = []

    for trial in manifest:
        if trial["cat"] != "new_note":
            continue

        if trial.get("rec_status") == "REC_ERROR":
            continue

        try:
            audio = load(trial)
            if audio is None:
                continue

            result, _, _ = analyze_recording(audio, trial["played"])

            if result["status"] == "OK" and np.isfinite(result["buzz_level"]):
                levels.append(result["buzz_level"])

        except Exception:
            log.write(
                f"[pass 1: {trial['name']}]\n"
                f"{traceback.format_exc()}\n"
            )

    session_level = float(np.median(levels)) if levels else None

    level_text = (
        f"{session_level:.3e}"
        if session_level is not None
        else "unavailable; using floor + 12 dB"
    )

    print(f"Session buzz level (from {len(levels)} trials): {level_text}")

    rows = []
    counts = {}

    csv_path = f"results_new_note_{output_tag}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(RESULT_HEADER)

        for trial in manifest:
            control = trial["cat"] == "new_note_control"
            notes = list(trial.get("notes", []))

            overlaps = harmonic_overlap(trial["played"])
            if overlaps:
                notes.append(
                    "HARMONIC_NEAR_MOTOR_BAND:" + ";".join(overlaps)
                )

            result = empty_result()
            tone_env = None
            motor_env = None

            if trial.get("rec_status") == "REC_ERROR":
                result["status"] = "REC_ERROR"
            else:
                try:
                    audio = load(trial)

                    if audio is None:
                        result["status"] = "MISSING_WAV"
                    else:
                        result, tone_env, motor_env = analyze_recording(
                            audio,
                            trial["played"],
                            control=control,
                            session_level=session_level
                        )

                except Exception as exc:
                    result = empty_result()
                    result["status"] = "ANALYSIS_ERROR"

                    notes.append(f"{type(exc).__name__}: {exc}")

                    log.write(
                        f"[{trial['name']}] ANALYSIS_ERROR\n"
                        f"{traceback.format_exc()}\n"
                    )

            if result["gate_note"]:
                notes.append(result["gate_note"])

            save_plot(
                plot_dir, trial["name"],
                result, tone_env, motor_env
            )

            status = result["status"]
            counts[status] = counts.get(status, 0) + 1

            if control:
                target = "no buzz"
                met = {
                    "CONTROL_QUIET": "Y",
                    "CONTROL_FALSE_TRIGGER": "N"
                }.get(status, "N/A")

            elif status == "OK":
                target = f"{TARGET_MS:.0f}"
                met = "Y" if result["delay_raw_ms"] <= TARGET_MS else "N"

            else:
                target = "N/A"
                met = "N/A"

            writer.writerow([
                trial.get("order", ""),
                trial.get("timestamp", ""),
                trial["name"],
                trial["label"],
                trial["cat"],
                trial["note"],
                trial["rep"],
                fmt(trial["base"]),
                fmt(trial["cents"]),
                fmt(trial["played"]),
                status,
                fmt(result["tone_onset_ms"]),
                fmt(result["vib_ms"]),
                fmt(result["delay_raw_ms"]),
                fmt(result["cal_tone_ms"]),
                fmt(result["cal_motor_ms"]),
                fmt(result["delay_ms"]),
                fmt(result["tone_snr_db"], 1),
                fmt(result["motor_snr_db"], 1),
                f"{result['gate']:.3e}" if np.isfinite(result["gate"]) else "NaN",
                result["ft_count"] if control else "N/A",
                fmt(result["ft_first_ms"]),
                result["off_count"] if control else "N/A",
                target,
                met,
                ANALYSIS_VERSION,
                FIRMWARE_VERSION,
                " | ".join(notes)
            ])

            f.flush()

            rows.append({
                "label": trial["label"],
                "status": status,
                "raw": result["delay_raw_ms"],
                "corr": result["delay_ms"],
                "off": result["off_count"]
            })

            log.write(
                f"[{trial['name']}] {status} "
                f"raw={fmt(result['delay_raw_ms'])} "
                f"ft={result['ft_count']} "
                f"off={result['off_count']} notes={notes}\n"
            )

    summary_path = f"summary_new_note_{output_tag}.csv"
    write_summary(rows, summary_path)

    print("Trial statuses:", counts)
    print(f"Saved {csv_path}, {summary_path} and plots in {plot_dir}/")

    return session_level, len(levels), counts


def manifest_from_wavs(wav_dir, original_csv=None):
    lookup = {}

    for label, category, note, base, cents, _ in CONDITIONS:
        lookup[label] = (category, note, base, cents)

    previous = {}

    if original_csv and os.path.exists(original_csv):
        with open(original_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                previous[row["Trial ID"]] = (
                    row.get("Run order", ""),
                    row.get("Timestamp", ""),
                    row.get("Status", "")
                )

    manifest = []

    for filename in os.listdir(wav_dir):
        if not filename.startswith("Trial_") or not filename.endswith(".wav"):
            continue

        name = filename[:-4]

        try:
            label, rep = name[len("Trial_"):].rsplit("_", 1)
            rep = int(rep)
        except ValueError:
            continue

        if label not in lookup:
            print(f"  Skipping {name}: condition '{label}' is not configured")
            continue

        category, note, base, cents = lookup[label]
        order, stamp, _ = previous.get(name, ("", "", ""))

        manifest.append({
            "name": name,
            "label": label,
            "cat": category,
            "note": note,
            "rep": rep,
            "base": base,
            "cents": cents,
            "played": base * 2 ** (cents / 1200.0),
            "order": int(order) if str(order).isdigit() else "",
            "timestamp": stamp
        })

    present = {trial["name"] for trial in manifest}

    for name, (order, stamp, status) in previous.items():
        if name in present or status != "REC_ERROR":
            continue

        label, rep = name[len("Trial_"):].rsplit("_", 1)

        if label not in lookup:
            continue

        category, note, base, cents = lookup[label]

        manifest.append({
            "name": name,
            "label": label,
            "cat": category,
            "note": note,
            "rep": int(rep),
            "base": base,
            "cents": cents,
            "played": base * 2 ** (cents / 1200.0),
            "rec_status": "REC_ERROR",
            "order": int(order) if str(order).isdigit() else "",
            "timestamp": stamp
        })

    manifest.sort(
        key=lambda trial: (
            trial["order"] if trial["order"] != "" else 10 ** 6,
            trial["name"]
        )
    )

    return manifest


def warm_up_audio():
    print(f"Warming up the audio stream ({WARMUP_CALLS} silent calls)...")

    silence = np.zeros(int(WARMUP_LEN * SR), dtype=np.float32)
    ok = 0

    for i in range(WARMUP_CALLS):
        try:
            rec = sd.playrec(
                silence,
                samplerate=SR,
                channels=1,
                device=DEVICE_CONFIG
            )

            sd.wait()

            ok += rec is not None and len(rec) == len(silence)

        except Exception as exc:
            print(f"  Warm-up call {i + 1} failed: {exc}")

        time.sleep(0.5)

    time.sleep(1.0)

    print(f"Warm-up complete: {ok}/{WARMUP_CALLS} full buffers")
    return ok


def record_session(trials, wav_dir, log):
    os.makedirs(wav_dir, exist_ok=True)

    for trial in trials:
        print(
            f"[{trial['order']}/{len(trials)}] Recording {trial['name']} "
            f"({trial['note']}, {trial['cents']:+.0f} cents = "
            f"{trial['played']:.2f} Hz)"
        )

        time.sleep(INTER_TRIAL_GAP)

        trial["timestamp"] = datetime.now().isoformat(timespec="seconds")
        trial["notes"] = []

        try:
            padded = np.concatenate((
                np.zeros(int(PRE_PAD * SR)),
                make_stimulus(trial["played"]),
                np.zeros(int(POST_PAD * SR))
            )).astype(np.float32)

            rec = sd.playrec(
                padded,
                samplerate=SR,
                channels=1,
                device=DEVICE_CONFIG
            )

            sd.wait()

            audio = np.asarray(rec, dtype=np.float64).flatten()

            if len(audio) != len(padded) or not np.all(np.isfinite(audio)):
                raise RuntimeError(
                    f"Incomplete recording: {len(audio)} samples, "
                    f"expected {len(padded)}"
                )

            wav_path = os.path.join(
                wav_dir, f"{trial['name']}.wav"
            )

            sf.write(wav_path, audio, SR, subtype="FLOAT")
            trial["rec_status"] = "RECORDED"

        except Exception as exc:
            trial["rec_status"] = "REC_ERROR"
            trial["notes"].append(f"{type(exc).__name__}: {exc}")

            log.write(
                f"[{trial['name']}] REC_ERROR\n"
                f"{traceback.format_exc()}\n"
            )

            print(f"  Recording failed: {exc}")

    return trials


def main():
    parser = argparse.ArgumentParser(
        description="Measure new-note to vibration delay"
    )

    parser.add_argument("--reanalyze", metavar="WAV_DIR")
    parser.add_argument("--orig-csv", metavar="CSV")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow recording over an existing session with the same RUN_TAG"
    )

    args = parser.parse_args()

    params = {
        key: value
        for key, value in globals().items()
        if key.isupper() and isinstance(
            value, (int, float, tuple, str, list)
        )
    }

    meta = {
        "run_tag": RUN_TAG,
        "analysis_version": ANALYSIS_VERSION,
        "firmware_version": FIRMWARE_VERSION,
        "started": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "params": params,
        "status_codes": STATUS_DESCRIPTIONS
    }

    if args.reanalyze:
        source_name = os.path.basename(os.path.normpath(args.reanalyze))
        output_tag = f"{RUN_TAG}_reanalysis_of_{source_name}"

        manifest = manifest_from_wavs(
            args.reanalyze, args.orig_csv
        )

        wav_dir = args.reanalyze

        meta.update(
            mode="reanalyze",
            source_wavs=os.path.abspath(args.reanalyze)
        )

    else:
        output_tag = RUN_TAG
        wav_dir = f"trial_wavs_{RUN_TAG}"

        if os.path.isdir(wav_dir) and not args.overwrite:
            has_wavs = any(
                filename.endswith(".wav")
                for filename in os.listdir(wav_dir)
            )

            if has_wavs:
                print(
                    f"Stopped: '{wav_dir}' already has recordings. "
                    "Change RUN_TAG, or add --overwrite to replace them."
                )
                sys.exit(1)

        print(sd.query_devices())

        manifest = build_trials()

        meta.update(
            mode="record",
            sounddevice=sd.__version__,
            devices={
                "mic": str(sd.query_devices(MIC_DEV)["name"]),
                "spk": str(sd.query_devices(SPK_DEV)["name"])
            },
            trial_order=[trial["name"] for trial in manifest],
            warmup_ok_calls=warm_up_audio()
        )

    log_path = f"run_log_{output_tag}.txt"
    meta_path = f"run_meta_{output_tag}.json"

    with open(log_path, "w", encoding="utf-8") as log:
        try:
            if not args.reanalyze:
                print(
                    f"Recording {len(manifest)} trials "
                    f"(seed {SHUFFLE_SEED})"
                )

                record_session(manifest, wav_dir, log)

            print("Analyzing recordings...")

            level, n_level, counts = analyze_session(
                manifest, wav_dir, output_tag, log
            )

            cal_data = {}

            for freq, offsets in _CALIBRATION_CACHE.items():
                cal_data[f"{freq:.2f}"] = [
                    round(value * 1000, 3) for value in offsets
                ]

            meta.update(
                session_buzz_level=level,
                session_level_n_trials=n_level,
                status_counts=counts,
                calibration_ms=cal_data,
                finished=datetime.now().isoformat(timespec="seconds")
            )

        finally:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=str)

    print("Finished.")


if __name__ == "__main__":
    main()