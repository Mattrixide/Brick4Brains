#pragma once
#include "config.h"

// ============================================================
// Motor — DRV8871 dual driver + trapezoidal turn profiling
// ============================================================

struct TrapezoidalProfile {
    float theta_target;   // degrees (absolute)
    float omega_max;      // dps
    float alpha;          // dps/s (angular acceleration)
    float t_accel;
    float t_cruise;
    float t_total;
    float omega_peak;
    float direction;      // +1 or -1

    void compute(float target_deg, float max_omega, float max_alpha);
    float getOmega(float t) const;
    float getTheta(float t) const;
    bool isDone(float t) const { return t >= t_total; }
};

class Motor {
public:
    void begin();

    // Direct drive: normalized -1.0 to 1.0
    void drive(float throttle, float steering);

    // Set individual motor powers: -1.0 to 1.0
    void setLeft(float power);
    void setRight(float power);

    // Stop both motors (coast)
    void stop();

    // Brake both motors (active brake)
    void brake();

private:
    void setMotor(int ch_in1, int ch_in2, float power);
};

// PID controller for angular velocity
class TurnPID {
public:
    TurnPID(float kp, float ki, float kd, float i_limit);

    float compute(float error, float dt);
    void reset();

private:
    float _kp, _ki, _kd;
    float _i_limit;
    float _integral = 0.0f;
    float _prev_error = 0.0f;
    bool _first = true;
};
