import argparse
import csv
import os
import sys
import numpy as np
import sounddevice as sd

RATE = 48000
BUF_DUR = 2.0
AMP = 0.5

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
OFFSETS = [0, 2, -2, 4, -4, 6, -6, 10, -10]

def midi_to_name(m):
    octave = (m // 12) - 1
    return f"{NOTES[m % 12]}{octave}"

def get_tones():
    # builds the 351 tones in sheet order (midi 55 to 93)
    lst = []
    for m in range(55, 94):
        n_str = midi_to_name(m)
        for off in OFFSETS:
            sign = f"+{off}" if off > 0 else str(off)
            fname = f"{n_str}_{m}_{off}.wav"
            f_hz = 440.0 * (2.0 ** ((m - 69 + (off / 100.0)) / 12.0))
            lst.append({
                "idx": len(lst),
                "note": n_str,
                "midi": m,
                "offset": off,
                "freq": f_hz,
                "file": fname,
                "desc": f"{n_str} (MIDI {m}, {sign} cents) -> {f_hz:.2f} Hz"
            })
    return lst

def start_tone(hz):
    # use an exact integer number of cycles so loop=True doesn't click at the seam
    cycles = max(1, int(round(hz * BUF_DUR)))
    dur = cycles / hz
    t = np.linspace(0, dur, int(round(RATE * dur)), endpoint=False)
    wave = AMP * np.sin(2.0 * np.pi * hz * t)
    
    fade = min(int(0.015 * RATE), len(wave))
    wave[:fade] *= np.linspace(0.0, 1.0, fade)

    sd.play(wave.astype(np.float32), samplerate=RATE, loop=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", "-s", type=int, default=0, help="resume index")
    parser.add_argument("--limit", "-l", type=int, default=None, help="test limit")
    parser.add_argument("--out", "-o", type=str, default="tuner_benchmark_results.csv")
    args = parser.parse_args()

    tones = get_tones()
    total = len(tones)

    if args.start < 0 or args.start >= total:
        print(f"bad start index. must be 0 to {total - 1}")
        sys.exit(1)

    end = total if args.limit is None else min(total, args.start + args.limit)
    has_file = os.path.isfile(args.out) and os.path.getsize(args.out) > 0

    print(f"loaded {total} tones. starting at {args.start}.")
    print("type reading (e.g. -3 or 3) while tone plays, hit enter for blank, ctrl+c to bail.\n")

    i = args.start
    with open(args.out, mode="a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not has_file:
            w.writerow(["index", "filename", "note", "midi", "expected_cents", "tuner_reading"])
            f.flush()

        try:
            for i in range(args.start, end):
                t = tones[i]
                print(f"[{i}/{total - 1}] playing: {t['desc']}")
                start_tone(t["freq"])

                ans = input("  cents reading (enter for blank): ").strip()
                sd.stop()

                w.writerow([i, t["file"], t["note"], t["midi"], t["offset"], ans])
                f.flush()

        except KeyboardInterrupt:
            sd.stop()
            print(f"\naborted at index {i}. resume with: python tuner_benchmark.py --start {i}")
            sys.exit(0)

    print("\nall done! dump the tuner_reading column into column F.")

if __name__ == "__main__":
    main()