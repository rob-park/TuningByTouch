import numpy as np
import soundfile as sf
import os
Sample_Rate = 48000
Chunk_Size = 1024
TONES_DIR = os.path.join(os.path.dirname(__file__), "tones")
file_path = os.path.join(TONES_DIR, "A4_+0c_440.00Hz.wav")
audio, sr = sf.read(file_path)
mid = len(audio)//2
chunk = audio[mid: mid + Chunk_Size]
print(sr)
print(len(audio))
print(len(chunk))