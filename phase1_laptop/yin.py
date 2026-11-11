import numpy as np
import soundfile as sf
import os
Sample_Rate = 48000
Chunk_Size = 1024
#block 2:
min_freq = 180
max_freq =2000
tau_min = int(Sample_Rate/ max_freq)
tau_max = int(Sample_Rate / min_freq)
window = Chunk_Size - tau_max
TONES_DIR = os.path.join(os.path.dirname(__file__), "tones")
file_path = os.path.join(TONES_DIR, "A4_+0c_440.00Hz.wav")
audio, sr = sf.read(file_path)
mid = len(audio)//2
chunk = audio[mid: mid + Chunk_Size]


#block 2

def difference(chunk):
    d = np.zeros(tau_max + 1)
    for tau in range(1,tau_max +1):
        original = chunk[0:window]
        shifted = chunk[tau : tau + window]
        d[tau] = np.sum((original - shifted) ** 2)
    return d

#2b
def normalize(d):
    d_prime = np.zeros(len(d))
    d_prime[0] = 1.0
    running = 0.0
    for tau in range(1,len(d)):
        running += d[tau]
        if running ==0:
            d_prime[tau] = 1.0
        else:
            d_prime[tau] = d[tau] *tau /running
    return d_prime


#2c
def find_dip(d_prime, threshold =0.15):
    upper_bound = min(tau_max, len(d_prime) -1)
    for tau in range(tau_min,upper_bound+1):
        if d_prime[tau] <threshold:
            while tau +1<=upper_bound and d_prime[tau+1] <d_prime[tau]:
                tau +=  1
            return tau
    return -1


#2d
def interpolate(d, best):
    if best <= 0 or best >= len(d) -1:
        return float(best)
    s0 = d[best-1]
    s1 = d[best]
    s2 = d[best+1]
    denom = s0 -2 *s1 +s2
    if denom ==0:
        return float(best)
    return best +0.5*(s0-s2)/denom


def yin_pitch(chunk):
    d = difference(chunk)
    d_prime = normalize(d)
    best = find_dip(d_prime)
    if best == -1:
        return -1,1.0
    better = interpolate(d,best)
    freq=Sample_Rate/better
    confidence = d_prime[best]
    return freq, confidence

freq, conf = yin_pitch(chunk)
print(freq, conf)