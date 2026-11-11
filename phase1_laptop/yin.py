import librosa
import numpy as np
import soundfile as sf
import os
import csv
Sample_Rate = 48000
Chunk_Size = 1024
#block 2:
min_freq = 180
max_freq =2000
tau_min = int(Sample_Rate/ max_freq)
tau_max = int(Sample_Rate / min_freq)
window = Chunk_Size - tau_max
TONES_DIR = os.path.join(os.path.dirname(__file__), "tones")


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

if __name__ == "__main__":
    csv_path = os.path.join(TONES_DIR, "answer_key.csv")
    results = []
    errors = []
    lib_errors = []
    no_pitch = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wav_path = os.path.join(TONES_DIR, row["filename"])
            audio,sr = sf.read(wav_path)
            mid = len(audio) //2
            chunk = audio[mid: mid +Chunk_Size]
            freq, conf = yin_pitch(chunk)
            lib_freq = librosa.yin(
                chunk, 
                fmin=min_freq, 
                fmax=max_freq, 
                sr=Sample_Rate, 
                frame_length=Chunk_Size, 
                center=False, 
                trough_threshold=0.15
            )[0]
            if freq == -1:
                no_pitch += 1
                continue
            true_freq = float(row["true_freq_hz"])
            cents_error = 1200 * np.log2(freq / true_freq)
            lib_error = 1200 * np.log2(lib_freq / true_freq)
            errors.append(cents_error)
            lib_errors.append(lib_error)
            results.append(
                [row["filename"], true_freq, round(freq, 3), round(cents_error, 3), conf, round(lib_freq, 3), round(lib_error, 3)])
        errors = np.array(errors)
        lib_errors = np.array(lib_errors)


    total = len(errors)
    within_3 = np.sum(np.abs(errors) <= 3)
    print("Total:", total)
    print("Within 3 cents:", within_3)
    print("Percent:", within_3 / total * 100)
    print("Max error:", np.max(np.abs(errors)))
    print("Average error:", np.mean(errors))
    print("No pitch:", no_pitch)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "yin_python_results.csv")
    with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "true_freq_hz",
            "measured_freq_hz",
            "cents_error",
            "confidence",
            "librosa_freq_hz",
            "librosa_cents_error",
            ])
        writer.writerows(results)

    print(f"Saved {len(results)} rows to {out_csv}")
    lib_within_3 = np.sum(np.abs(lib_errors) <= 3)
    print("Librosa Within 3 cents:", lib_within_3)
    print("Librosa Percent:", lib_within_3 / total * 100)
    print("Librosa Max error:", np.max(np.abs(lib_errors)))
    print("Librosa Average error:", np.mean(lib_errors))

    print("Max diff b/t my YIN + librosa:", np.max(np.abs(errors - lib_errors)))