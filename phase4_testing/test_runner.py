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
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results_motors_off.csv")

REPEATS = 1
SILENCE_S = 1.0

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
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    time.sleep(2)
    
    try:
        with open(ANSWER_KEY, mode="r", encoding="utf-8") as f:
            tones = list(csv.DictReader(f))
            
        tones = tones[-3:]          # TEMP - A6 check
        
        header = ["filename", "note", "midi", "offset_cents", "true_freq_hz", "repeat", "t_s", "freq_hz", "cents", "confidence", "calc_us"]
        
        with open(RESULTS_PATH, mode="w", newline="", encoding="utf-8") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(header)
            
            total_count = 0
            for rep in range(1, REPEATS + 1):
                for tone in tones:
                    time.sleep(SILENCE_S)
                    readings = test_tone(ser, tone["filename"])
                    
                    for t, line in readings:
                        parts = line.split(",")
                        if len(parts) != 4:
                            continue
                        row = [
                            tone["filename"],
                            tone["note"],
                            tone["midi"],
                            tone["offset_cents"],
                            tone["true_freq_hz"],
                            rep,
                            t,
                            *parts
                        ]
                        writer.writerow(row)
                        
                    out_f.flush()
                    total_count += 1
                    
                    if total_count % 25 == 0:
                        print(f"Processed {total_count}/{len(tones) * REPEATS} tones...")
                        
        print("Test complete! Results saved to results_motors_off.csv")

    except KeyboardInterrupt:
        print("\nInterrupted by user, shutting down safely...")
    finally:
        ser.close()
        print("Serial port closed.")

if __name__ == "__main__":
    main()