from yin import yin_pitch, Chunk_Size
import librosa
import numpy as np
import re
import csv
import os
import difflib
IOWA_DIR = os.path.join(os.path.dirname(__file__),"iowa")
files = sorted([f for f in os.listdir(IOWA_DIR) if f.endswith("aiff")])

correct = 0
total = 0
out_of_range = 0
no_pitch = 0
rows = []

for f in files:
    path = os.path.join(IOWA_DIR, f)
    audio, sr = librosa.load(path, sr=48000, mono=True)
    
    segments = librosa.effects.split(audio, top_db=30)
    segments = [s for s in segments if (s[1] - s[0]) / 48000 >= 0.1]
    
    part = f.split(".")[4]
    ends = re.findall(r"[A-G][b#]?\d", part)
    first = librosa.note_to_midi(ends[0])
    last  = librosa.note_to_midi(ends[-1])
    expected = list(range(first, last + 1))
    expected = [e for e in expected if e <= 93]
    
    if not expected:
        continue

    measured = []
    for start, end in segments:
        mid = (start + end) // 2
        chunk = audio[mid: mid + Chunk_Size]
        if len(chunk) < Chunk_Size:
            no_pitch += 1
            continue
        freq, conf = yin_pitch(chunk)
        if freq == -1:
            no_pitch += 1
            continue
        m = int(round(69 + 12 * np.log2(freq / 440.0)))
        if m <= 93:
            measured.append(m)
            rows.append([f, 'N/A', m, round(freq, 2)])
        else:
            out_of_range += 1

    sm = difflib.SequenceMatcher(None, expected, measured)
    matched = sum(block.size for block in sm.get_matching_blocks())
    correct += matched
    total += len(expected)

results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)
out_csv = os.path.join(results_dir, "midi_evaluation_results.csv")

with open(out_csv, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "expected_midi", "measured_midi", "measured_freq_hz"])
    writer.writerows(rows)

pct = (correct / total * 100) if total > 0 else 0.0
print(f"\nCorrect: {correct} / {total} ({pct:.2f}%)")
print(f"Out of range (> A6): {out_of_range}")
print(f"No pitch detected: {no_pitch}")
print(f"Saved {len(rows)} rows to {out_csv}")
for r in rows:
    if r[1] != r[2]:
        print(r)