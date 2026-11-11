// INMP441 + ESP32 mic test
// Wiring: VDD->3.3V, GND->GND, L/R->GND, SCK->GPIO14, WS->GPIO15, SD->GPIO32
// See serial plotter for results

#include <driver/i2s.h>

#define I2S_SCK  14
#define I2S_WS   15
#define I2S_SD   32
#define I2S_PORT I2S_NUM_0

const int SAMPLE_RATE = 16000;
const int BUF_LEN = 256;
int32_t samples[BUF_LEN];

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
    .dma_buf_len = BUF_LEN,
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

  if (i2s_driver_install(I2S_PORT, &config, 0, NULL) != ESP_OK) {
    Serial.println("I2S driver install FAILED");
    while (true) delay(1000);
  }
  i2s_set_pin(I2S_PORT, &pins);
  i2s_zero_dma_buffer(I2S_PORT);
}

void loop() {
  size_t bytesRead = 0;
  i2s_read(I2S_PORT, samples, sizeof(samples), &bytesRead, portMAX_DELAY);
  int n = bytesRead / sizeof(int32_t);
  if (n == 0) return;

  long long sum = 0;
  for (int i = 0; i < n; i++) {
    samples[i] >>= 14;
    sum += samples[i];
  }
  int32_t mean = sum / n;

  int32_t peak = 0;
  for (int i = 0; i < n; i++) {
    int32_t v = abs(samples[i] - mean);
    if (v > peak) peak = v;
  }

  Serial.println(peak);
}