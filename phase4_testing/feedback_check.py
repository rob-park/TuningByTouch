import argparse
import csv
import os
import sys
import threading
import time
from collections import Counter
from datetime import datetime

import numpy as np
import serial
import sounddevice as sd


RUN_TAG = "fb_run1"
FIRMWARE_VERSION = "settle=2"

SERIAL_PORT = "COM6"
BAUD = 115200
SPK_DEV = 3
SHUFFLE_SEED = 20261116

SR = 48000
A4_HZ = 440.0
IN_TUNE_WINDOW = 5.0

MIDI_LOW, MIDI_HIGH = 55, 93
OFFSETS = [2, -2, 4, -4, 6, -6, 10, -10]
TRIALS = 5

PRE_SILENCE = 0.5
TONE_DUR = 1.5
POST_SILENCE = 0.7
GAP = 0.3
FADE = 0.01
VOLUME = 0.5

ONSET_SEARCH = 1.5
STEADY_FROM = 0.5
END_GUARD = 0.05

NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]

STATE_NAMES = {
    0: "In tune",
    1: "Sharp",
    2: "Flat"
}

RESULTS = f"results_feedback_{RUN_TAG}.csv"
RAW = f"raw_serial_feedback_{RUN_TAG}.csv"
PASTE = f"paste_feedback_{RUN_TAG}.csv"
SUMMARY = f"summary_feedback_{RUN_TAG}.csv"

RESULT_HEADER = [
    "Sheet row", "Trial", "Note", "MIDI #", "Offset (cents)",
    "Target freq (Hz)", "Expected feedback", "Device feedback",
    "Correct?", "Flicker count", "Group", "Status",
    "Lines in steady window", "Share of steady window in that state",
    "Device mean cents (steady)", "Timestamp", "Firmware"
]


def note_name(midi):
    octave = midi // 12 - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"


def build_tones():
    tones = []
    row = 6

    for trial in range(1, TRIALS + 1):
        for midi in range(MIDI_LOW, MIDI_HIGH + 1):
            base_freq = A4_HZ * 2 ** ((midi - 69) / 12)

            for off in OFFSETS:
                freq = base_freq * 2 ** (off / 1200)

                if abs(off) < IN_TUNE_WINDOW:
                    expected = "In tune"
                elif off > 0:
                    expected = "Sharp"
                else:
                    expected = "Flat"

                group = "Clear" if abs(off) in (2, 10) else "Near cutoff"

                tones.append({
                    "row": row,
                    "trial": trial,
                    "midi": midi,
                    "note": note_name(midi),
                    "offset": off,
                    "freq": freq,
                    "expected": expected,
                    "group": group
                })

                row += 1

    return tones


def play_order(tones):
    rng = np.random.default_rng(SHUFFLE_SEED)
    shuffled = []

    for trial in range(1, TRIALS + 1):
        block = [tone for tone in tones if tone["trial"] == trial]
        indices = rng.permutation(len(block))

        for idx in indices:
            shuffled.append(block[idx])

    return shuffled


def make_audio(freq):
    n_samples = int(SR * TONE_DUR)
    t = np.arange(n_samples) / SR
    tone = np.sin(2 * np.pi * freq * t)

    fade_len = int(FADE * SR)
    tone[:fade_len] *= np.linspace(0, 1, fade_len)
    tone[-fade_len:] *= np.linspace(1, 0, fade_len)

    before = np.zeros(int(SR * PRE_SILENCE))
    after = np.zeros(int(SR * POST_SILENCE))

    audio = np.concatenate((before, tone, after))
    return (audio * VOLUME).astype(np.float32)


class SerialReader(threading.Thread):
    def __init__(self, port, baud):
        super().__init__(daemon=True)

        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = baud
        self.ser.timeout = 0.2

        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()

        self.lines = []
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            try:
                raw = self.ser.readline()
            except Exception:
                continue

            if not raw:
                continue

            now = time.perf_counter()
            text = raw.decode("utf-8", errors="replace").strip()
            parts = text.split(",")

            if len(parts) != 5:
                continue

            try:
                freq = float(parts[0])
                cents = float(parts[1])
                state = int(parts[4])
            except ValueError:
                continue

            with self.lock:
                self.lines.append((now, freq, cents, state, text))

    def since(self, t0):
        with self.lock:
            return [line for line in self.lines if line[0] >= t0]

    def trim(self, keep_after):
        with self.lock:
            self.lines = [
                line for line in self.lines
                if line[0] >= keep_after
            ]

    def stop(self):
        self.running = False

        try:
            self.ser.close()
        except Exception:
            pass


def judge(lines, play_start):
    result = {
        "status": "OK",
        "feedback": "",
        "flicker": "",
        "n": 0,
        "share": "",
        "cents": ""
    }

    pre_end = play_start + PRE_SILENCE * 0.8
    before = [line for line in lines if line[0] < pre_end]

    if any(line[3] != 0 for line in before[-5:]):
        result["status"] = "NOT_SILENT_BEFORE"
        return result

    search_end = play_start + PRE_SILENCE + ONSET_SEARCH

    heard = [
        line for line in lines
        if line[1] > 0 and play_start <= line[0] <= search_end
    ]

    if not heard:
        result["status"] = "NO_PITCH"
        return result

    onset = heard[0][0]
    note_end = onset + TONE_DUR - END_GUARD

    held = [
        line for line in lines
        if onset <= line[0] <= note_end
    ]

    steady_start = onset + STEADY_FROM
    steady = [line for line in held if line[0] >= steady_start]

    if len(steady) < 5:
        result["status"] = "TOO_FEW_LINES"
        return result

    counts = Counter(line[3] for line in steady)
    state, n_state = counts.most_common(1)[0]

    changes = 0
    for prev, curr in zip(held, held[1:]):
        if prev[3] != curr[3]:
            changes += 1

    flicker = changes

    if state != 0 and held[0][3] == 0 and changes >= 1:
        flicker -= 1

    cents_values = [
        line[2] for line in steady
        if line[1] > 0
    ]

    mean_cents = (
        f"{np.mean(cents_values):.2f}"
        if cents_values else ""
    )

    result.update({
        "feedback": STATE_NAMES.get(state, f"state {state}"),
        "flicker": max(0, flicker),
        "n": len(steady),
        "share": f"{n_state / len(steady):.2f}",
        "cents": mean_cents
    })

    return result


def load_done():
    if not os.path.exists(RESULTS):
        return {}

    with open(RESULTS, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {int(row["Sheet row"]): row for row in reader}


def export(tones):
    done = load_done()

    with open(PASTE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Sheet row",
            "Device feedback (column G)",
            "Flicker count (column I)"
        ])

        for tone in tones:
            row = done.get(tone["row"], {})

            writer.writerow([
                tone["row"],
                row.get("Device feedback", ""),
                row.get("Flicker count", "")
            ])

    groups = [
        ("Clear", "+-2, +-10", 0.95),
        ("Near cutoff", "+-4, +-6", 0.80)
    ]

    with open(SUMMARY, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Group", "Offsets", "Tones done", "Valid",
            "Correct", "% correct", "Target",
            "Tones with flicker", "Excluded (status:count)"
        ])

        for group, offsets, target in groups:
            rows = [
                done[tone["row"]]
                for tone in tones
                if tone["group"] == group and tone["row"] in done
            ]

            valid = [row for row in rows if row["Status"] == "OK"]
            correct = sum(row["Correct?"] == "Y" for row in valid)

            flicker_count = sum(
                1 for row in valid
                if row["Flicker count"] not in ("", "0")
            )

            excluded = Counter(
                row["Status"] for row in rows
                if row["Status"] != "OK"
            )

            accuracy = (
                f"{100 * correct / len(valid):.1f}%"
                if valid else "N/A"
            )

            excluded_text = "; ".join(
                f"{status}:{count}"
                for status, count in excluded.items()
            ) or "none"

            writer.writerow([
                group,
                offsets,
                len(rows),
                len(valid),
                correct,
                accuracy,
                f"{target:.0%}",
                flicker_count,
                excluded_text
            ])

    print(f"Wrote {PASTE} and {SUMMARY}")


def main():
    parser = argparse.ArgumentParser(
        description="Feedback Check boundary test"
    )

    parser.add_argument(
        "--max",
        type=int,
        help="only run this many new tones (for a quick check)"
    )

    parser.add_argument(
        "--export",
        action="store_true",
        help="only rebuild the paste and summary files"
    )

    args = parser.parse_args()
    tones = build_tones()

    if args.export:
        export(tones)
        return

    done = load_done()
    todo = [
        tone for tone in play_order(tones)
        if tone["row"] not in done
    ]

    if args.max:
        todo = todo[:args.max]

    duration = PRE_SILENCE + TONE_DUR + POST_SILENCE + GAP
    minutes = len(todo) * duration / 60

    print(
        f"{len(done)} tones already done, {len(todo)} to run "
        f"(about {minutes:.0f} min)."
    )

    if not todo:
        export(tones)
        return

    print(
        f"Opening {SERIAL_PORT}... "
        "(close Arduino Serial Monitor if this fails)"
    )

    reader = SerialReader(SERIAL_PORT, BAUD)
    reader.start()

    time.sleep(2.0)

    if not reader.since(time.perf_counter() - 2.0):
        print(
            "No data lines from the device in 2 s. "
            "Check the port, the cable and that the "
            "firmware is printing. Stopping."
        )

        reader.stop()
        sys.exit(1)

    silence = np.zeros(int(0.3 * SR), dtype=np.float32)
    sd.play(silence, SR, device=SPK_DEV)
    sd.wait()

    new_file = not os.path.exists(RESULTS)
    raw_new = not os.path.exists(RAW)

    with open(RESULTS, "a", newline="", encoding="utf-8") as results_f, \
         open(RAW, "a", newline="", encoding="utf-8") as raw_f:

        result_writer = csv.writer(results_f)
        raw_writer = csv.writer(raw_f)

        if new_file:
            result_writer.writerow(RESULT_HEADER)

        if raw_new:
            raw_writer.writerow([
                "Sheet row",
                "Time from playback start (s)",
                "Device line"
            ])

        try:
            for i, tone in enumerate(todo, 1):
                reader.trim(time.perf_counter() - 1.0)
                play_start = time.perf_counter()

                try:
                    audio = make_audio(tone["freq"])
                    sd.play(audio, SR, device=SPK_DEV)
                    sd.wait()

                    lines = reader.since(play_start - PRE_SILENCE)
                    result = judge(lines, play_start)

                except Exception as e:
                    lines = []
                    result = {
                        "status": f"ERROR: {type(e).__name__}: {e}",
                        "feedback": "",
                        "flicker": "",
                        "n": 0,
                        "share": "",
                        "cents": ""
                    }

                feedback = result["feedback"]

                if feedback:
                    correct = "Y" if feedback == tone["expected"] else "N"
                else:
                    correct = ""

                result_writer.writerow([
                    tone["row"],
                    tone["trial"],
                    tone["note"],
                    tone["midi"],
                    tone["offset"],
                    f"{tone['freq']:.2f}",
                    tone["expected"],
                    feedback,
                    correct,
                    result["flicker"],
                    tone["group"],
                    result["status"],
                    result["n"],
                    result["share"],
                    result["cents"],
                    datetime.now().isoformat(timespec="seconds"),
                    FIRMWARE_VERSION
                ])

                for line in lines:
                    raw_writer.writerow([
                        tone["row"],
                        f"{line[0] - play_start:.3f}",
                        line[4]
                    ])

                results_f.flush()
                raw_f.flush()

                flag = ""

                if result["status"] != "OK" or correct != "Y":
                    flag = f"  <-- {result['status']} {correct}"

                print(
                    f"[{len(done) + i}/{len(tones)}] "
                    f"row {tone['row']}: trial {tone['trial']} "
                    f"{tone['note']} {tone['offset']:+d}c -> "
                    f"{feedback or '-'} "
                    f"(expected {tone['expected']}), "
                    f"flicker {result['flicker']}{flag}"
                )

                time.sleep(GAP)

        except KeyboardInterrupt:
            print("\nStopped. Progress is saved; run again to continue.")

        finally:
            reader.stop()

    export(tones)


if __name__ == "__main__":
    main()