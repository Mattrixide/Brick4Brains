#pragma once

// ============================================================
// B4B IMU Combat Firmware — Configuration
// ============================================================

// -- WiFi --
#define WIFI_SSID       "YOUR_SSID"
#define WIFI_PASSWORD   "YOUR_PASSWORD"
#define MDNS_HOSTNAME   "esp32wifi"        // resolves as esp32wifi.local

// -- UDP Ports --
#define CMD_PORT        4210               // receive motor commands from PC
#define TELEMETRY_PORT  4211               // send IMU telemetry to PC

// -- Motor Driver (DRV8871, 2-pin per motor) --
// DRV8871: IN1 high + IN2 low = forward, IN1 low + IN2 high = reverse
//          Both low = coast, both high = brake
#define MOTOR_L_IN1     25
#define MOTOR_L_IN2     26
#define MOTOR_R_IN1     27
#define MOTOR_R_IN2     14

#define PWM_FREQ        25000              // 25 kHz (above audible)
#define PWM_RESOLUTION  8                  // 8-bit (0-255)

// PWM channel assignments (ESP32 LEDC)
#define PWM_CH_L_IN1    0
#define PWM_CH_L_IN2    1
#define PWM_CH_R_IN1    2
#define PWM_CH_R_IN2    3

// -- IMU (ISM330DHCX via Qwiic I2C) --
#define IMU_SDA         21
#define IMU_SCL         22
#define IMU_I2C_FREQ    400000             // 400 kHz Fast Mode
#define IMU_ADDR        0x6B               // SparkFun default

// -- Control Loop --
#define CONTROL_RATE_HZ 416                // match IMU ODR
#define CONTROL_PERIOD_US (1000000 / CONTROL_RATE_HZ)  // ~2404 us

// -- Failsafe --
#define CMD_TIMEOUT_MS  200                // stop motors if no command for 200ms
#define WATCHDOG_SEC    3                  // hardware watchdog

// -- Telemetry --
#define TELEMETRY_RATE_HZ  100             // 100 Hz telemetry output
#define TELEMETRY_PERIOD_MS (1000 / TELEMETRY_RATE_HZ)

// -- Gyro Calibration --
#define GYRO_CALIB_SAMPLES 416             // ~1 second at 416 Hz
#define GYRO_DEADZONE_DPS  0.3f            // noise rejection threshold

// -- Turn Control (Feedforward + PID) --
#define TURN_KV         0.05f              // V per dps (tune empirically)
#define TURN_KS         0.30f              // static friction (normalized 0-1)
#define TURN_KP         0.03f              // PID proportional
#define TURN_KI         0.005f             // PID integral
#define TURN_KD         0.0005f            // PID derivative
#define TURN_I_LIMIT    0.5f               // integral windup clamp
#define TURN_OMEGA_MAX  720.0f             // max angular velocity (dps)
#define TURN_ALPHA      3600.0f            // angular acceleration (dps/s)

// -- Command Modes --
#define MODE_DIRECT     0                  // pass-through throttle/steering
#define MODE_GYRO_TURN  1                  // gyro-assisted turn to heading delta
