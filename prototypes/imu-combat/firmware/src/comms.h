#pragma once
#include <WiFiUdp.h>
#include "config.h"

// ============================================================
// Comms — UDP command receiver + telemetry sender
// ============================================================

// Command packet from PC (5 or 8 bytes)
struct Command {
    int16_t throttle;       // -32767 to 32767 (forward positive)
    int16_t steering;       // -32767 to 32767 (right positive)
    uint8_t buttons;        // bitmask
    uint8_t mode;           // 0=direct, 1=gyro-turn (byte 5, optional)
    int16_t heading_delta;  // 0.01 degree units (bytes 6-7, optional)
    bool valid;
};

// Telemetry packet to PC (20 bytes)
struct Telemetry {
    float heading;          // integrated heading (degrees)
    float gyro_z;           // raw gyro Z (dps)
    float accel_x;          // mg
    float accel_y;          // mg
    uint32_t timestamp;     // millis()
};

class Comms {
public:
    bool begin();

    // Check for incoming command (non-blocking)
    bool receiveCommand(Command &cmd);

    // Send telemetry to last known PC address
    void sendTelemetry(const Telemetry &tel);

    // Time since last valid command (ms)
    unsigned long timeSinceLastCmd() const;

    // Is the link timed out?
    bool isTimedOut() const { return timeSinceLastCmd() > CMD_TIMEOUT_MS; }

private:
    WiFiUDP _cmdUdp;
    WiFiUDP _telUdp;
    IPAddress _pcAddr;
    uint16_t _pcPort = 0;
    bool _pcKnown = false;
    unsigned long _lastCmdTime = 0;
};
