#pragma once
#include <SparkFun_ISM330DHCX.h>
#include "config.h"

// ============================================================
// IMU — ISM330DHCX gyro/accel via I2C
// ============================================================

class IMU {
public:
    bool begin();
    bool calibrateGyro();       // block for ~1s while stationary
    bool read();                // read latest gyro + accel

    // Gyro Z with bias removed and deadzone applied (dps)
    float gyroZ() const { return _gyroZ; }
    // Raw accel (mg)
    float accelX() const { return _accelX; }
    float accelY() const { return _accelY; }
    float accelZ() const { return _accelZ; }

    // Integrated heading (degrees, accumulates)
    float heading() const { return _heading; }
    void resetHeading(float deg = 0.0f) { _heading = deg; }

    // Integrate gyro into heading (call at control rate)
    void integrateHeading(float dt);

    bool isCalibrated() const { return _calibrated; }
    bool isSaturated() const { return _saturated; }

private:
    SparkFun_ISM330DHCX _sensor;
    sfe_ism_data_t _gyroRaw;
    sfe_ism_data_t _accelRaw;

    float _biasZ = 0.0f;
    float _gyroZ = 0.0f;      // bias-corrected, deadzone-filtered (dps)
    float _accelX = 0.0f;
    float _accelY = 0.0f;
    float _accelZ = 0.0f;
    float _heading = 0.0f;    // integrated heading (degrees)
    bool _calibrated = false;
    bool _saturated = false;
};
