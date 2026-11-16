import os
import time
import csv
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import butter, lfilter, hilbert
import matplotlib.pyplot as plt
RUN_TAG = "final"
SR = 48000
REC_TIME = 4.0

MIC_DEV = 1   # MME C920 Webcam Mic
SPK_DEV = 3   # MME Edifier Speakers
devCfg = (MIC_DEV, SPK_DEV)

PLOTS_DIR = "trial_plots"
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

WAVS_DIR = "trial_wavs"
if not os.path.exists(WAVS_DIR):
    os.makedirs(WAVS_DIR)

MOTOR_LOW = 1180.0
MOTOR_HIGH = 1250.0

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def gen_pitch_stim(f1, f2, dur1=1.5, dur2=1.5):
    t1 = np.linspace(0, dur1, int(SR * dur1), endpoint=False)
    t2 = np.linspace(0, dur2, int(SR * dur2), endpoint=False)
    
    p1 = 2 * np.pi * f1 * t1
    p2 = p1[-1] + 2 * np.pi * f2 * t2
    
    waveform = np.concatenate([np.sin(p1), np.sin(p2)])
    
    fade = int(0.01 * SR)
    win = np.ones_like(waveform)
    win[:fade] = np.linspace(0, 1, fade)
    win[-fade:] = np.linspace(1, 0, fade)
    
    return waveform * win

def save_plot(t_axis, env, thresh, jump_t, tone_onset_t, vib_t, trialName, delay_val, status):
    plt.figure(figsize=(10, 4))
    plt.plot(t_axis, env, label='motor env', color='navy', alpha=0.7)
    plt.axhline(thresh, color='r', linestyle='--', label='thresh')
    plt.axvline(tone_onset_t, color='purple', linestyle=':', label='tone onset')
    plt.axvline(jump_t, color='g', label='jump')
    if vib_t is not None:
        plt.axvline(vib_t, color='orange', linestyle='-.', label='motor start')
    plt.title(f"{trialName} [{status}] | delay: {delay_val}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{trialName}.png"))
    plt.close()

def analyze(audioData, tone_freq, trialName):
    tone_filtered = bandpass_filter(audioData, tone_freq - 50.0, tone_freq + 50.0, SR)
    tone_env = np.abs(hilbert(tone_filtered))
    
    t_axis = np.linspace(0, len(audioData) / SR, len(audioData))
    
    # Start tone onset search at 0.5s (end of silent padding) to skip startup thumps
    transient_skip = int(0.5 * SR)
    search_slice = tone_env[transient_skip:]
    
    steady_lvl = np.percentile(tone_env, 90)
    cross_idx = np.where(search_slice >= 0.5 * steady_lvl)[0]
    
    if len(cross_idx) == 0:
        print(f"ERROR [{trialName}]: tone never showed up in mic data")
        save_plot(t_axis, np.zeros_like(t_axis), 0, 0, 0, None, trialName, "N/A", "NO_TONE")
        return np.nan, np.nan, np.nan
        
    tone_onset_sec = (transient_skip + cross_idx[0]) / SR
    
    # Plausibility window: expect tone onset around 0.84s (0.5s padding + ~0.34s speaker/air lag)
    if not (0.6 <= tone_onset_sec <= 1.2):
        print(f"ERROR [{trialName}]: tone onset at {tone_onset_sec:.3f}s outside expected 0.6-1.2s window!")
        save_plot(t_axis, np.zeros_like(t_axis), 0, 0, tone_onset_sec, None, trialName, "N/A", "EARLY_TONE_NOISE")
        return np.nan, np.nan, np.nan
        
    jump_time_sec = tone_onset_sec + 1.5

    filtered = bandpass_filter(audioData, MOTOR_LOW, MOTOR_HIGH, SR)
    env = np.abs(hilbert(filtered))
    
    base_slice = env[int(1.8 * SR):int(2.2 * SR)]
    base_mean = np.mean(base_slice)
    base_std = np.std(base_slice)
    # Note: absolute floor of 0.00005 overrides 3*sigma when baseline is extremely quiet
    thresh = max(base_mean + 3.0 * base_std, 0.00005)
    
    jump_idx = int(jump_time_sec * SR)
    check_start = max(int(0.5 * SR), jump_idx - int(0.5 * SR))
    
    pre_exceed = env[check_start:jump_idx] > thresh
    min_pre_samp = int(0.002 * SR) # 2ms of sustained signal
    pre_buzz_found = False
    for i in range(len(pre_exceed) - min_pre_samp):
        if np.all(pre_exceed[i : i + min_pre_samp]):
            pre_buzz_found = True
            break

    if pre_buzz_found:
        print(f"WARNING [{trialName}]: motor active before jump, rejecting")
        save_plot(t_axis, env, thresh, jump_time_sec, tone_onset_sec, None, trialName, "N/A", "ALREADY_BUZZING")
        return jump_time_sec * 1000.0, np.nan, np.nan

    min_samp = int(0.005 * SR)
    exceed = env[jump_idx:] > thresh
    
    vib_time = None
    for i in range(len(exceed) - min_samp):
        if np.all(exceed[i : i + min_samp]):
            vib_time = (jump_idx + i) / SR
            break
            
    if vib_time is None:
        print(f"WARNING [{trialName}]: motor start not found, silent fail safeguard triggered")
        save_plot(t_axis, env, thresh, jump_time_sec, tone_onset_sec, None, trialName, "N/A", "NO_MOTOR")
        return jump_time_sec * 1000.0, np.nan, np.nan
        
    delay_ms = (vib_time - jump_time_sec) * 1000.0
    
    save_plot(t_axis, env, thresh, jump_time_sec, tone_onset_sec, vib_time, trialName, f"{delay_ms:.1f}ms", "OK")
    return jump_time_sec * 1000.0, vib_time * 1000.0, delay_ms

def main():
    print(sd.query_devices())
    print(f"device config: {devCfg}")
    print("starting pitch change tests...")
    
    f_base = 440.0
    f_20c = f_base * (2 ** (20.0 / 1200.0))
    f_6c = f_base * (2 ** (6.0 / 1200.0))
    g3_base = 196.00
    g3_shift = g3_base * (2 ** (20.0 / 1200.0))
    a6_base = 1760.00
    a6_shift = a6_base * (2 ** (20.0 / 1200.0))
    
    trials = []
    for i in range(1, 6):
        trials.append((i, f_base, f_20c, 20.0, "pitch"))
    for i in range(1, 6):
        trials.append((i, f_base, f_6c, 6.0, "pitch_report"))
    
    extra_cfgs = [
        (1, g3_base, g3_shift, 20.0), (2, g3_base, g3_shift, 20.0),
        (3, a6_base, a6_shift, 20.0), (4, a6_base, a6_shift, 20.0),
        (5, f_base, f_20c, 20.0), (6, f_base, f_20c, 20.0),
        (7, f_base, f_20c, 20.0), (8, f_base, f_20c, 20.0),
        (9, f_base, f_20c, 20.0), (10, f_base, f_20c, 20.0)
    ]
    for t_num, sf_hz, ef_hz, cents in extra_cfgs:
        trials.append((t_num, sf_hz, ef_hz, cents, "pitch_extra"))

    csv_name = "results_pitch_change.csv"
    with open(csv_name, mode="w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Trial", "Start freq (Hz)", "End freq (Hz)", "Jump (cents)", "Jump time in recording (ms)", "Vibration start (ms)", "Delay (ms)", "Target (ms)", "Met target?"])
        
        for t_num, f1, f2, cents, cat in trials:
            trial_name = f"Trial_{cat}_{t_num}"
            print(f"running {trial_name} ({f1} -> {f2} Hz)...")
            time.sleep(2.0)
            
            stim = gen_pitch_stim(f1, f2)
            pre_pad = np.zeros(int(0.5 * SR))
            post_pad = np.zeros(int(1.0 * SR))
            padded_stim = np.concatenate([pre_pad, stim, post_pad])
            
            audio_rec = sd.playrec(padded_stim, samplerate=SR, channels=1, device=devCfg)
            sd.wait()
            
            audio_flat = audio_rec.flatten()
            
            # Save raw WAV file with FLOAT subtype for exact bit-for-bit reproducibility
            sf.write(os.path.join(WAVS_DIR, f"{trial_name}.wav"), audio_flat, SR, subtype="FLOAT")
            
            jump_ms, vib_ms, delay = analyze(audio_flat, f1, trial_name)
            
            if np.isnan(delay):
                print(f"-> {trial_name} failed, wrote NaN")
                w.writerow([t_num, f"{f1:.2f}", f"{f2:.2f}", f"{cents:.2f}", f"{jump_ms:.2f}", "NaN", "NaN", "N/A", "FAILED"])
                f.flush()
                continue
            
            target = 100.0
            met = "Y" if delay <= target else "N"
            if "report" in cat:
                met = "Report only"
                
            w.writerow([t_num, f"{f1:.2f}", f"{f2:.2f}", f"{cents:.2f}", f"{jump_ms:.2f}", f"{vib_ms:.2f}", f"{delay:.2f}", target if met != "Report only" else "N/A", met])
            f.flush()
            
    print("all done!")

if __name__ == "__main__":
    main()