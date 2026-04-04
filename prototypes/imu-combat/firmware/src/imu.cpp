#include "imu.h"
#include <Wire.h>

bool IMU::begin() {
    Wire.begin(IMU_SDA, IMU_SCL);
    Wire.setClock(IMU_I2C_FREQ);

    if (!_sensor.begin(Wire, IMU_ADDR)) {
        Serial.println("[imu] ISM330DHCX not found");
        return false;
    }

    // Reset to known state
    _sensor.deviceReset();
    while (!_sensor.getDeviceReset()) {
        delay(1);
    }

    // Gyro: 416 Hz, +/-2000 dps
    _sensor.setGyroDataRate(ISM_GY_ODR_416Hz);
    _sensor.setGyroFullScale(ISM_2000dps);
    _sensor.setGyroFilterLP1();
    _sensor.setGyroLP1Bandwidth(ISM_MEDIUM);

    // Accel: 416 Hz, +/-4g
    _sensor.setAccelDataRate(ISM_XL_ODR_416Hz);
    _sensor.setAccelFullScale(ISM_4g);

    Serial.println("[imu] ISM330DHCX initialized (416Hz, +/-2000dps)");
    return true;
}

bool IMU::calibrateGyro() {
    Serial.println("[imu] Calibrating gyro — hold still...");

    float sum = 0.0f;
    int count = 0;

    for (int i = 0; i < GYRO_CALIB_SAMPLES; i++) {
        // Wait for data ready
        unsigned long start = micros();
        while (!_sensor.checkStatus()) {
            if (micros() - start > 10000) break;  // 10ms timeout
        }
        _sensor.getGyro(&_gyroRaw);
        sum += _gyroRaw.zData;  // mdps
        count++;
    }

    if (count < GYRO_CALIB_SAMPLES / 2) {
        Serial.println("[imu] Calibration failed — not enough samples");
        return false;
    }

    _biasZ = sum / count;
    _calibrated = true;

    Serial.printf("[imu] Gyro bias Z: %.2f mdps (%d samples)\n", _biasZ, count);
    return true;
}

bool IMU::read() {
    if (!_sensor.checkStatus()) {
        return false;
    }

    _sensor.getGyro(&_gyroRaw);
    _sensor.getAccel(&_accelRaw);

    // Bias-correct and convert mdps -> dps
    float raw_dps = (_gyroRaw.zData - _biasZ) / 1000.0f;

    // Dead-zone threshold
    if (fabsf(raw_dps) < GYRO_DEADZONE_DPS) {
        raw_dps = 0.0f;
    }

    // Saturation detection
    _saturated = fabsf(raw_dps) > 1900.0f;

    _gyroZ = raw_dps;
    _accelX = _accelRaw.xData;  // mg
    _accelY = _accelRaw.yData;
    _accelZ = _accelRaw.zData;

    return true;
}

void IMU::integrateHeading(float dt) {
    if (!_saturated) {
        _heading += _gyroZ * dt;
    }
    // During saturation: hold heading (don't integrate garbage)
}
