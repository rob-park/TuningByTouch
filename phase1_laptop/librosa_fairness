import os, csv
import numpy as np
import soundfile as sf
import librosa

Sample_Rate = 48000
min_freq = 180
max_freq = 2000
TONES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tones")
csv_path = os.path.join(TONES_DIR, "answer_key.csv")

errors = []
with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        audio, sr = sf.read(os.path.join(TONES_DIR, row["filename"]))
        f0 = librosa.yin(audio, fmin=min_freq, fmax=max_freq, sr=Sample_Rate,
                         frame_length=2048, trough_threshold=0.15)
        freq = np.median(f0)
        true_freq = float(row["true_freq_hz"])
        errors.append(1200 * np.log2(freq / true_freq))

errors = np.array(errors)
print("realistic librosa check: (whole tone, 2048 frames, median)")
print("Within 3 cents:", np.sum(np.abs(errors) <= 3), "/", len(errors))
print("Percent:", np.mean(np.abs(errors) <= 3) * 100)
print("Max error:", np.max(np.abs(errors)))
print("Avg error:", np.mean(errors))