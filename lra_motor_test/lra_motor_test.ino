#include <Wire.h>
#include <Haptic_Driver.h>

Haptic_Driver motor1;
Haptic_Driver motor2;

void setup() {
  Serial.begin(115200);
  delay(400);

  Wire.begin(8, 9);
  Wire1.begin(17, 18);
  
  Serial.println("booting up haptic drivers...");

  if (!motor1.begin(Wire)) {
    Serial.println("err: driver 1 not responding");
    while(1) { delay(500); }
  }

  if (!motor2.begin(Wire1)) {
    Serial.println("err: driver 2 not responding");
    while(1) { delay(500); }
  }

  motor1.defaultMotor();
  motor1.enableFreqTrack(false);
  motor1.setOperationMode(DRO_MODE);

  motor2.defaultMotor();
  motor2.enableFreqTrack(false);
  motor2.setOperationMode(DRO_MODE);
  
  Serial.println("both drivers ready");
}

void loop() {
  motor1.setVibrate(25);
  delay(500);
  motor1.setVibrate(0);
  delay(200);

  motor2.setVibrate(25);
  delay(500);
  motor2.setVibrate(0);
  delay(500);
}