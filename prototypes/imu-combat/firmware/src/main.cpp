// ============================================================
// B4B IMU Combat Firmware
//
// ESP32 + ISM330DHCX + DRV8871 motor drivers
// Receives UDP commands from PC, provides IMU telemetry,
// and executes gyro-assisted turns with feedforward + PID.
//
// Modes:
//   0 (direct):    pass-through throttle/steering
//   1 (gyro-turn): execute precise turn using trapezoidal
//                   motion profile with gyro feedback at 416 Hz
// ============================================================

#include <Arduino.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include <esp_task_wdt.h>

#include "config.h"
#include "imu.h"
#include "motor.h"
#include "comms.h"

// -- Globals --
IMU imu;
Motor motor;
Comms comms;

// Turn state
enum TurnState { TURN_IDLE, TURN_ACTIVE, TURN_DONE };
TurnState turnState = TURN_IDLE;
TrapezoidalProfile turnProfile;
TurnPID turnPid(TURN_KP, TURN_KI, TURN_KD, TURN_I_LIMIT);
float turnStartHeading = 0.0f;
unsigned long turnStartTime = 0;

// Timing
unsigned long lastControlTime = 0;
unsigned long lastTelemetryTime = 0;

// Direct command cache (for sending between control ticks)
float cmdThrottle = 0.0f;
float cmdSteering = 0.0f;

// ============================================================
// WiFi setup
// ============================================================
void setupWiFi() {
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);  // disable power saving for low latency
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    Serial.print("[wifi] Connecting");
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 40) {
        delay(500);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[wifi] Connected: %s\n", WiFi.localIP().toString().c_str());

        if (MDNS.begin(MDNS_HOSTNAME)) {
            Serial.printf("[wifi] mDNS: %s.local\n", MDNS_HOSTNAME);
        }
    } else {
        Serial.println("\n[wifi] FAILED — running without network");
    }
}

// ============================================================
// Turn execution (called at control rate)
// ============================================================
void updateTurn(float dt) {
    if (turnState != TURN_ACTIVE) return;

    float elapsed = (micros() - turnStartTime) / 1000000.0f;

    // Check if profile is done
    if (turnProfile.isDone(elapsed)) {
        motor.brake();
        delay(10);
        motor.stop();
        turnState = TURN_DONE;
        return;
    }

    // Setpoints from motion profile
    float omega_setpoint = turnProfile.getOmega(elapsed);   // dps
    float theta_setpoint = turnProfile.getTheta(elapsed);    // degrees

    // Measured values
    float omega_measured = imu.gyroZ();                      // dps
    float theta_measured = imu.heading() - turnStartHeading; // degrees

    // Feedforward: predict motor command from desired angular velocity
    float ff = TURN_KV * omega_setpoint;
    if (fabsf(omega_setpoint) > 0.1f) {
        ff += TURN_KS * ((omega_setpoint > 0) ? 1.0f : -1.0f);
    }

    // PID on angular velocity error
    float vel_error = omega_setpoint - omega_measured;
    float pid_out = turnPid.compute(vel_error, dt);

    // Outer position correction (keeps angle on track)
    float pos_error = theta_setpoint - theta_measured;
    float pos_correction = 0.01f * pos_error;

    // Combined output
    float output = constrain(ff + pid_out + pos_correction, -1.0f, 1.0f);

    // Spin in place: left and right opposite directions
    motor.setLeft(-output);
    motor.setRight(output);
}

// ============================================================
// Start a gyro-assisted turn
// ============================================================
void startTurn(float delta_deg) {
    turnProfile.compute(delta_deg, TURN_OMEGA_MAX, TURN_ALPHA);
    turnPid.reset();
    turnStartHeading = imu.heading();
    turnStartTime = micros();
    turnState = TURN_ACTIVE;

    Serial.printf("[turn] Start %.1f deg (est %.0f ms)\n",
                  delta_deg, turnProfile.t_total * 1000);
}

// ============================================================
// Process incoming command
// ============================================================
void processCommand(const Command &cmd) {
    if (cmd.mode == MODE_GYRO_TURN && turnState != TURN_ACTIVE) {
        // heading_delta is in 0.01 degree units
        float delta_deg = cmd.heading_delta / 100.0f;
        if (fabsf(delta_deg) > 0.5f) {  // ignore tiny turns
            startTurn(delta_deg);
        }
    } else if (cmd.mode == MODE_DIRECT) {
        // Cancel any active turn
        if (turnState == TURN_ACTIVE) {
            turnState = TURN_IDLE;
            turnPid.reset();
        }
        // Normalize int16 -> float [-1, 1]
        cmdThrottle = cmd.throttle / 32767.0f;
        cmdSteering = cmd.steering / 32767.0f;
    }
}

// ============================================================
// Send telemetry
// ============================================================
void sendTelemetry() {
    Telemetry tel;
    tel.heading   = imu.heading();
    tel.gyro_z    = imu.gyroZ();
    tel.accel_x   = imu.accelX();
    tel.accel_y   = imu.accelY();
    tel.timestamp = millis();
    comms.sendTelemetry(tel);
}

// ============================================================
// Setup
// ============================================================
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n=== B4B IMU Combat Firmware ===");

    // Motor init (before WiFi to ensure motors are stopped)
    motor.begin();
    motor.stop();

    // WiFi
    setupWiFi();

    // UDP
    comms.begin();

    // IMU
    if (!imu.begin()) {
        Serial.println("[FATAL] IMU init failed — halting");
        while (true) { delay(1000); }
    }
    imu.calibrateGyro();

    // Hardware watchdog
    esp_task_wdt_init(WATCHDOG_SEC, true);
    esp_task_wdt_add(NULL);

    lastControlTime = micros();
    lastTelemetryTime = millis();

    Serial.println("[main] Ready — waiting for commands");
}

// ============================================================
// Main loop
// ============================================================
void loop() {
    // Feed watchdog
    esp_task_wdt_reset();

    // 1. Check for incoming commands (non-blocking)
    Command cmd;
    if (comms.receiveCommand(cmd)) {
        processCommand(cmd);
    }

    // 2. Read IMU at control rate (~416 Hz)
    unsigned long now_us = micros();
    if (now_us - lastControlTime >= CONTROL_PERIOD_US) {
        float dt = (now_us - lastControlTime) / 1000000.0f;
        lastControlTime = now_us;

        if (imu.read()) {
            imu.integrateHeading(dt);
        }

        // 3. Execute control based on state
        if (comms.isTimedOut()) {
            // Failsafe: no commands for 200ms -> stop
            motor.stop();
            if (turnState == TURN_ACTIVE) {
                turnState = TURN_IDLE;
                turnPid.reset();
            }
            cmdThrottle = 0;
            cmdSteering = 0;
        } else if (turnState == TURN_ACTIVE) {
            // Gyro-assisted turn in progress
            updateTurn(dt);
        } else {
            // Direct drive mode
            motor.drive(cmdThrottle, cmdSteering);
        }
    }

    // 4. Send telemetry at 100 Hz
    unsigned long now_ms = millis();
    if (now_ms - lastTelemetryTime >= TELEMETRY_PERIOD_MS) {
        lastTelemetryTime = now_ms;
        sendTelemetry();
    }
}
