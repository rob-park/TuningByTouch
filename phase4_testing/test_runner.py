import os
import time
import csv
import serial
import soundfile as sf
import sounddevice as sd

PORT = "COM6"   # <- set to the port that works today
BAUD = 115200
TONES_DIR = os.path.join(os.path.dirname(__file__), "..", "phase1_laptop", "tones")
ANSWER_KEY = os.path.join(TONES_DIR, "answer_key.csv")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results_motors_on_rerun.csv")
SILENCE_S = 1.0

# (midi, offset_cents, repeat number it replaces)
REDO = [(70, -2, 1), (70, 2, 1), (76, -2, 1), (91, -10, 4), (56, -6, 5), (56, 6, 5)]

def test_tone(ser, filename):
    audio, sr = sf.read(os.path.join(TONES_DIR, filename))
    readings = []
    ser.reset_input_buffer()
    sd.play(audio, sr)
    t0 = time.time()
    while time.time() - t0 < 3.0:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            readings.append((round(time.time() - t0, 3), line))
    sd.stop()
    return readings

def main():
    with open(ANSWER_KEY, encoding="utf-8") as f:
        key = {(int(float(r["midi"])), float(r["offset_cents"])): r for r in csv.DictReader(f)}

    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    time.sleep(2)
    header = ["filename", "note", "midi", "offset_cents", "true_freq_hz", "repeat",
              "t_s", "freq_hz", "cents", "confidence", "calc_us", "state"]
    try:
        with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as out_f:
            w = csv.writer(out_f)
            w.writerow(header)
            for midi, off, rep in REDO:
                tone = key.get((midi, float(off)))
                if tone is None:
                    print(f"not found in answer key: {midi} {off}")
                    continue
                time.sleep(SILENCE_S)
                readings = test_tone(ser, tone["filename"])
                good = 0
                for t, line in readings:
                    parts = line.split(",")
                    if len(parts) != 5:
                        continue
                    good += 1
                    w.writerow([tone["filename"], tone["note"], tone["midi"], tone["offset_cents"],
                                tone["true_freq_hz"], rep, t, *parts])
                out_f.flush()
                print(f"{tone['filename']} rep {rep}: {good} valid lines")
        print("Done:", os.path.basename(RESULTS_PATH))
    finally:
        ser.close()

if __name__ == "__main__":
    main()