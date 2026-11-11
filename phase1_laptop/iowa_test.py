from yin import yin_pitch, Chunk_Size
import librosa
import numpy as np
import re
import csv

IOWA_DIR = os.path.join(os.path.dirname(__file__),"iowa")
files = sorted([f for f in os.listdir(IOWA_DIR) if f.endswith("aiff")])
print(len(files))