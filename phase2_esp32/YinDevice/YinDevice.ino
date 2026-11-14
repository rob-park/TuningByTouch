#include <driver/i2s.h>
#include <math.h>

#define I2S_SCK  14
#define I2S_WS   15
#define I2S_SD   13
#define I2S_PORT I2S_NUM_0

const int   SAMPLE_RATE = 48000;
const int   CHUNK_SIZE  = 1024;
const float MIN_FREQ    = 180.0;
const float MAX_FREQ    = 2000.0;
const int   TAU_MIN     = SAMPLE_RATE / MAX_FREQ;
const int   TAU_MAX     = SAMPLE_RATE / MIN_FREQ;
const int   WINDOW      = CHUNK_SIZE - TAU_MAX;
const float THRESHOLD   = 0.15;
const float SILENCE_RMS = 60000.0f;

int32_t raw[CHUNK_SIZE];
float   buf[CHUNK_SIZE];
float   d[TAU_MAX + 1];
float   dPrime[TAU_MAX + 1];

void calc_diff() {
    for (int tau = 1; tau <= TAU_MAX; tau++) {
        d[tau] = 0.0f;
        for (int j = 0; j < WINDOW; j++) {
            float diff = buf[j] - buf[j + tau];
            d[tau] += diff * diff;
        }
    }
}

void normalize_d() {
    dPrime[0] = 1.0f;
    float running_sum = 0.0f;
    for (int tau = 1; tau <= TAU_MAX; tau++) {
        running_sum += d[tau];
        if (running_sum == 0.0f) {
            dPrime[tau] = 1.0f;
        } else {
            dPrime[tau] = d[tau] * tau / running_sum;
        }
    }
}

int find_dip() {
    for (int tau = TAU_MIN; tau <= TAU_MAX; tau++) {
        if (dPrime[tau] < THRESHOLD) {
            while (tau + 1 <= TAU_MAX && dPrime[tau + 1] < dPrime[tau]) {
                tau++;
            }
            return tau;
        }
    }
    return -1;
}

float interpolate_tau(int best) {
    if (best <= 0 || best >= TAU_MAX) return (float)best;
    
    float s0 = d[best - 1];
    float s1 = d[best];
    float s2 = d[best + 1];
    float denom = s0 - 2.0f * s1 + s2;
    
    if (denom == 0.0f) return (float)best;
    return best + 0.5f * (s0 - s2) / denom;
}

bool yin_pitch(float& freq_out, float& conf_out) {
    for (int i = 0; i < CHUNK_SIZE; i++) {
        buf[i] = (float)(raw[i] >> 8);
    }

    float mean = 0.0f;
    for (int i = 0; i < CHUNK_SIZE; i++) {
        mean += buf[i];
    }
    mean /= CHUNK_SIZE;
    
    float rms = 0.0f;
    for (int i = 0; i < CHUNK_SIZE; i++) {
        buf[i] -= mean;
        rms += buf[i] * buf[i];
    }
    rms = sqrtf(rms / CHUNK_SIZE);

    if (rms < SILENCE_RMS) return false;

    calc_diff();
    normalize_d();

    int best = find_dip();
    if (best == -1) return false;

    float better_tau = interpolate_tau(best);
    if (better_tau <= 0.0f) return false;

    freq_out = (float)SAMPLE_RATE / better_tau;
    conf_out = dPrime[best];
    return true;
}

void setup() {
    Serial.begin(115200);

    const i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 512,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    const i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };

    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
    i2s_start(I2S_PORT);
}

void loop() {
    size_t bytes_read = 0;
    i2s_read(I2S_PORT, &raw, sizeof(raw), &bytes_read, portMAX_DELAY);

    uint32_t t_start = micros();
    float freq = 0.0f;
    float conf = 1.0f;
    bool got_pitch = yin_pitch(freq, conf);
    uint32_t calc_us = micros() - t_start;

    if (got_pitch) {
        float semitones = 12.0f * log2f(freq / 440.0f);
        int nearest = (int)roundf(semitones);
        float cents = 100.0f * (semitones - nearest);
        Serial.printf("%.2f,%.2f,%.3f,%lu\n", freq, cents, conf, calc_us);
    } else {
        Serial.printf("-1,0.00,1.00,%lu\n", calc_us);
    }
}