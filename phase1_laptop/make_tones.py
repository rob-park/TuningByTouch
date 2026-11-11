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
print(Fade_Samples)