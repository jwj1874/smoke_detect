/*
#include <Servo.h>

Servo servoX;  // X축 서보 모터
Servo servoY;  // Y축 서보 모터

void setup() {
  Serial.begin(9600);
  servoX.attach(9);  // X축 서보 핀
  servoY.attach(10); // Y축 서보 핀
}

void loop() {
  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');  // 데이터 수신
    int commaIndex = data.indexOf(',');
    int centerX = data.substring(0, commaIndex).toInt();
    int centerY = data.substring(commaIndex + 1).toInt();

    // 중심 좌표에 따라 서보 제어
    servoX.write(map(centerX, 0, 720, 0, 180));  // X축 제어
    servoY.write(map(centerY, 0, 720, 0, 180));  // Y축 제어
  }
}
*/

#include <Servo.h>

// 서보모터 핀
Servo servoX; // X축 서보
Servo servoY; // Y축 서보

// 초기 서보 위치
int posX = 90;
int posY = 90;

void setup() {
  Serial.begin(9600);
  servoX.attach(9); // X축 서보 핀
  servoY.attach(10); // Y축 서보 핀
  servoX.write(posX);
  servoY.write(posY);
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n'); // 좌표 데이터 수신
    int commaIndex = data.indexOf(',');
    if (commaIndex > 0) {
      int centerX = data.substring(0, commaIndex).toInt();
      int centerY = data.substring(commaIndex + 1).toInt();

      // X, Y 값에 따라 서보모터 위치 조정
      posX = map(centerX, 0, 640, 45, 135); // 화면 좌표를 서보 각도로 매핑
      posY = map(centerY, 0, 480, 45, 135);

      servoX.write(posX);
      servoY.write(posY);
    }
  }
}
