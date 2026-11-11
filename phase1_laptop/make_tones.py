import numpy as np
import soundfile as sf
import os
import csv
Sample_Rate = 48000
Duration = 3
Num_Harmonics = 6
Volume = 0.3
Fade_Samples = int(0.010 * Sample_Rate)
Offsets = [0,2,-2,4,-4,6,-6,10,-10]
Note_Names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
TONES_DIR = os.path.join(os.path.dirname(__file__), "tones")
os.makedirs(TONES_DIR, exist_ok=True)

#Block 2
num_samples = Sample_Rate * Duration
t = np.arange(num_samples) / Sample_Rate
fade_in = np.linspace(0,1,Fade_Samples)
fade_out = np.linspace(1,0,Fade_Samples)
rows = []

#Block 3
count = 0
for midi in range(55,94):
    base_freq = (440 * 2**((midi-69)/12))
    note = Note_Names[midi % 12] + str(midi // 12-1)
    for offset in Offsets:
        freq = base_freq * 2 ** (offset / 1200) 
        count += 1
        wave = np.zeros(len(t)) #Block 4
        for k in range(1, Num_Harmonics +1):
            wave += (1/k) * np.sin(2 * np.pi * k * freq * t)
        wave = wave / np.max(np.abs(wave)) * Volume
        wave[:Fade_Samples] *= fade_in
        wave[-Fade_Samples:] *= fade_out
        filename = f"{note}_{offset:+d}c_{freq:.2f}Hz.wav"
        sf.write(os.path.join(TONES_DIR, filename), wave, Sample_Rate)
        rows.append([filename, note, midi, offset, round(freq, 2)])
        print(f"Note: {note} Offset: {offset} Frequency: {freq}")

#Block 5:
csv_path = os.path.join(TONES_DIR, "answer_key.csv")
os.path.join(TONES_DIR, "answer_key.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "note", "midi", "offset_cents", "true_freq_hz"])
    writer.writerows(rows)