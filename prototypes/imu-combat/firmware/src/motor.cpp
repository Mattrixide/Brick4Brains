#include "motor.h"
#include <Arduino.h>

// ============================================================
// TrapezoidalProfile
// ============================================================

void TrapezoidalProfile::compute(float target_deg, float max_omega, float max_alpha) {
    direction = (target_deg >= 0) ? 1.0f : -1.0f;
    theta_target = fabsf(target_deg);
    omega_max = max_omega;
    alpha = max_alpha;

    t_accel = omega_max / alpha;
    float theta_accel = 0.5f * alpha * t_accel * t_accel;

    if (2.0f * theta_accel >= theta_target) {
        // Triangular — never reach omega_max
        t_accel = sqrtf(theta_target / alpha);
        omega_peak = alpha * t_accel;
        t_cruise = 0;
    } else {
        // Full trapezoidal
        omega_peak = omega_max;
        float theta_cruise = theta_target - 2.0f * theta_accel;
        t_cruise = theta_cruise / omega_max;
    }
    t_total = 2.0f * t_accel + t_cruise;
}

float TrapezoidalProfile::getOmega(float t) const {
    if (t < 0) return 0;
    if (t < t_accel) return direction * alpha * t;
    if (t < t_accel + t_cruise) return direction * omega_peak;
    if (t < t_total) return direction * alpha * (t_total - t);
    return 0;
}

float TrapezoidalProfile::getTheta(float t) const {
    float sign = direction;
    if (t < 0) return 0;
    if (t < t_accel) {
        return sign * 0.5f * alpha * t * t;
    }
    float theta_a = 0.5f * alpha * t_accel * t_accel;
    if (t < t_accel + t_cruise) {
        return sign * (theta_a + omega_peak * (t - t_accel));
    }
    float dt_end = t_total - t;
    if (t < t_total) {
        return sign * (theta_target - 0.5f * alpha * dt_end * dt_end);
    }
    return sign * theta_target;
}

// ============================================================
// Motor (DRV8871)
// ============================================================

void Motor::begin() {
    // Configure LEDC PWM channels
    ledcSetup(PWM_CH_L_IN1, PWM_FREQ, PWM_RESOLUTION);
    ledcSetup(PWM_CH_L_IN2, PWM_FREQ, PWM_RESOLUTION);
    ledcSetup(PWM_CH_R_IN1, PWM_FREQ, PWM_RESOLUTION);
    ledcSetup(PWM_CH_R_IN2, PWM_FREQ, PWM_RESOLUTION);

    // Attach pins to channels
    ledcAttachPin(MOTOR_L_IN1, PWM_CH_L_IN1);
    ledcAttachPin(MOTOR_L_IN2, PWM_CH_L_IN2);
    ledcAttachPin(MOTOR_R_IN1, PWM_CH_R_IN1);
    ledcAttachPin(MOTOR_R_IN2, PWM_CH_R_IN2);

    stop();
    Serial.println("[motor] DRV8871 initialized (25kHz PWM)");
}

void Motor::setMotor(int ch_in1, int ch_in2, float power) {
    power = constrain(power, -1.0f, 1.0f);

    uint8_t pwm = (uint8_t)(fabsf(power) * 255);

    if (power > 0.001f) {
        ledcWrite(ch_in1, pwm);
        ledcWrite(ch_in2, 0);
    } else if (power < -0.001f) {
        ledcWrite(ch_in1, 0);
        ledcWrite(ch_in2, pwm);
    } else {
        // Coast (both low)
        ledcWrite(ch_in1, 0);
        ledcWrite(ch_in2, 0);
    }
}

void Motor::setLeft(float power) {
    setMotor(PWM_CH_L_IN1, PWM_CH_L_IN2, power);
}

void Motor::setRight(float power) {
    setMotor(PWM_CH_R_IN1, PWM_CH_R_IN2, power);
}

void Motor::drive(float throttle, float steering) {
    throttle = constrain(throttle, -1.0f, 1.0f);
    steering = constrain(steering, -1.0f, 1.0f);

    // Arcade-style mix: throttle + steering -> left/right
    float left  = throttle - steering;
    float right = throttle + steering;

    // Scale to keep within [-1, 1] without clipping
    float maxVal = max(fabsf(left), fabsf(right));
    if (maxVal > 1.0f) {
        left  /= maxVal;
        right /= maxVal;
    }

    setLeft(left);
    setRight(right);
}

void Motor::stop() {
    setLeft(0);
    setRight(0);
}

void Motor::brake() {
    // DRV8871: both high = brake
    ledcWrite(PWM_CH_L_IN1, 255);
    ledcWrite(PWM_CH_L_IN2, 255);
    ledcWrite(PWM_CH_R_IN1, 255);
    ledcWrite(PWM_CH_R_IN2, 255);
}

// ============================================================
// TurnPID
// ============================================================

TurnPID::TurnPID(float kp, float ki, float kd, float i_limit)
    : _kp(kp), _ki(ki), _kd(kd), _i_limit(i_limit) {}

float TurnPID::compute(float error, float dt) {
    _integral += error * dt;
    _integral = constrain(_integral, -_i_limit, _i_limit);

    float derivative = 0.0f;
    if (!_first) {
        derivative = (error - _prev_error) / dt;
    }
    _prev_error = error;
    _first = false;

    return _kp * error + _ki * _integral + _kd * derivative;
}

void TurnPID::reset() {
    _integral = 0.0f;
    _prev_error = 0.0f;
    _first = true;
}
