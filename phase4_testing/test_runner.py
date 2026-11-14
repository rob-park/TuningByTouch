import os
import time
import serial
import soundfile as sf
import sounddevice as sd

PORT = "COM6"
BAUD = 115200
TONES_DIR = os.path.join(os.path.dirname(__file__), "..", "phase1_laptop", "tones")

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
    
    for t, line in test_tone(ser, "A4_+0c_440.00Hz.wav"):
        print(t, line)
        
    ser.close()

if __name__ == "__main__":
    main()