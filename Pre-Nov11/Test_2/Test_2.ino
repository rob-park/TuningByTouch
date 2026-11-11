// YIN pitch detection: ESP32-S3 + INMP441
// Wiring: VDD->3.3V, GND->GND, L/R->GND, SCK->14, WS->15, SD->13
// Open Serial Monitor at 115200 baud - play steady tone near mic

#include <driver/i2s.h>
#include <math.h>

#define I2S_SCK  14
#define I2S_WS   15
#define I2S_SD   13
#define I2S_PORT I2S_NUM_0

const int   SAMPLE_RATE = 44100;
const int   BUF_LEN     = 1024;
const int   HALF        = BUF_LEN / 2;
const float THRESHOLD   = 0.15;
const float MIN_FREQ    = 180.0;
const float MAX_FREQ    = 3000.0;
const float SILENCE_RMS = 200.0;

int32_t raw[BUF_LEN];
float   buf[BUF_LEN];
float   yin[HALF];

const char* NOTE_NAMES[12] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};

void setup() {
  Serial.begin(115200);
  delay(500);

  i2s_config_t config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 512,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  i2s_pin_config_t pins = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };
  i2s_driver_install(I2S_PORT, &config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pins);
  i2s_zero_dma_buffer(I2S_PORT);
  Serial.println("YIN pitch detector ready");
}

float yinPitch() {
  int tauMin = SAMPLE_RATE / MAX_FREQ;
  int tauMax = min((int)(SAMPLE_RATE / MIN_FREQ), HALF - 1);

  yin[0] = 1.0;
  for (int tau = 1; tau <= tauMax; tau++) {
    float sum = 0;
    for (int i = 0; i < HALF; i++) {
      float d = buf[i] - buf[i + tau];
      sum += d * d;
    }
    yin[tau] = sum;
  }

  float running = 0;
  for (int tau = 1; tau <= tauMax; tau++) {
    running += yin[tau];
    yin[tau] = (running == 0) ? 1.0 : yin[tau] * tau / running;
  }

  int best = -1;
  for (int tau = tauMin; tau <= tauMax; tau++) {
    if (yin[tau] < THRESHOLD) {
      while (tau + 1 <= tauMax && yin[tau + 1] < yin[tau]) tau++;
      best = tau;
      break;
    }
  }
  if (best < 0) return -1;

  float betterTau = best;
  if (best > 1 && best < tauMax) {
    float s0 = yin[best - 1], s1 = yin[best], s2 = yin[best + 1];
    float denom = s0 - 2 * s1 + s2;
    if (denom != 0) betterTau = best + 0.5 * (s0 - s2) / denom;
  }
  return SAMPLE_RATE / betterTau;
}

void loop() {
  size_t bytesRead = 0;
  i2s_read(I2S_PORT, raw, sizeof(raw), &bytesRead, portMAX_DELAY);
  int n = bytesRead / sizeof(int32_t);
  if (n < BUF_LEN) return;

  unsigned long t0 = micros();

  float mean = 0;
  for (int i = 0; i < BUF_LEN; i++) { buf[i] = (float)(raw[i] >> 14); mean += buf[i]; }
  mean /= BUF_LEN;
  float rms = 0;
  for (int i = 0; i < BUF_LEN; i++) { buf[i] -= mean; rms += buf[i] * buf[i]; }
  rms = sqrt(rms / BUF_LEN); 

  if (rms < SILENCE_RMS) { Serial.println("..."); return; }

  float freq = yinPitch();
  unsigned long computeUs = micros() - t0;

  if (freq < 0) { Serial.println("no clear pitch"); return; }

  float semis   = 12.0 * log2(freq / 440.0);
  int   nearest = (int)round(semis);
  float cents   = 100.0 * (semis - nearest);
  int   midi    = 69 + nearest;
  const char* name = NOTE_NAMES[((midi % 12) + 12) % 12];
  int   octave  = midi / 12 - 1;

  Serial.printf("%s%d  %7.2f Hz  %+6.1f cents  (calc %lu us)\n",
                name, octave, freq, cents, computeUs);
}