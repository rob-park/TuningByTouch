#include <driver/i2s.h>

#define I2S_SCK 14
#define I2S_WS  15
#define I2S_SD  13

#define I2S_PORT I2S_NUM_0

static uint64_t accum_samples = 0;
static unsigned long t_start = 0;
static int run_idx = 1;

const i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 48000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 512,
    .use_apll = false,
    .fixed_mclk = 0
};

const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
};

void setup() {
  Serial.begin(115200);
  delay(500); 
  
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
  i2s_start(I2S_PORT);

  delay(500);
  int32_t garbage_bin[128];
  size_t bytes_got = 0;
  for(int i = 0; i < 10; i++) {
    i2s_read(I2S_PORT, &garbage_bin, sizeof(garbage_bin), &bytes_got, portMAX_DELAY);
  }

  accum_samples = 0;
  t_start = millis();
  Serial.println("\n--- Sample Rate Test Running ---");
}

void loop() {
  int32_t rx_buff[256];
  size_t bytes_got = 0;

  i2s_read(I2S_PORT, &rx_buff, sizeof(rx_buff), &bytes_got, portMAX_DELAY);

  if (bytes_got > 0) {
    accum_samples += (bytes_got / 4);
  }

  if (millis() - t_start >= 60000) {
    Serial.printf("Run %d: %lu samples\n", run_idx, (unsigned long)accum_samples);
    
    if (run_idx >= 5) {
      Serial.println("Done! Copy these into column B.");
      while(true) { delay(1000); } 
    }

    run_idx++;
    accum_samples = 0;
    t_start = millis();
  }
}