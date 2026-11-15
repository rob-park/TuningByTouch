import os
import time
import csv
import serial
import soundfile as sf
import sounddevice as sd

PORT = "COM6"
BAUD = 115200
TONES_DIR = os.path.join(os.path.dirname(__file__), "..", "phase1_laptop", "tones")
ANSWER_KEY = os.path.join(TONES_DIR, "answer_key.csv")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results_motors_off_rerun.csv")

SILENCE_S = 1.0

# quick rerun list for the ones that glitched earlier
REDO_TARGETS = [
    ("B4_+6c_495.60Hz.wav", 5),
    ("E4_+4c_330.39Hz.wav", 3)
]

def test_tone(ser, filename):
    audio, sr = sf.read(os.path.join(TONES_DIR, filename))
    readings = []
    ser.reset_input_buffer()
    sd.play(audio, sr) # make sure volume/speaker is on!
    t0 = time.time()
    while time.time() - t0 < 3.0:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            readings.append((round(time.time() - t0, 3), line))
    sd.stop()
    return readings

def main():
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    time.sleep(2) # let serial settle
    
    try:
        with open(ANSWER_KEY, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # map filename to row so we can look it up fast
            toneMap = {r["filename"]: r for r in reader}
            
        header = ["filename", "note", "midi", "offset_cents", "true_freq_hz", "repeat", "t_s", "freq_hz", "cents", "confidence", "calc_us"]
        
        with open(RESULTS_PATH, mode="w", newline="", encoding="utf-8") as out_file:
            writer = csv.writer(out_file)
            writer.writerow(header)
            
            cnt = 0
            for fname, rep_num in REDO_TARGETS:
                if fname not in toneMap:
                    print(f"skipping {fname}, not in csv")
                    continue
                
                t_info = toneMap[fname]
                print(f"running {fname} with repeat {rep_num}...")
                
                time.sleep(SILENCE_S)
                readings = test_tone(ser, fname)
                
                for t, line in readings:
                    parts = line.split(",")
                    if len(parts) != 4:
                        continue # bad read line
                    row = [
                        t_info["filename"],
                        t_info["note"],
                        t_info["midi"],
                        t_info["offset_cents"],
                        t_info["true_freq_hz"],
                        rep_num, # keep original repeat num
                        t,
                        *parts
                    ]
                    writer.writerow(row)
                    
                out_file.flush()
                cnt += 1
                
        print(f"Done! saved to {RESULTS_PATH}")

    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        ser.close()
        print("closed serial")

if __name__ == "__main__":
    main()