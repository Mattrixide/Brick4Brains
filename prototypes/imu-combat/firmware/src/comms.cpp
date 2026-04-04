#include "comms.h"
#include <WiFi.h>

bool Comms::begin() {
    if (!_cmdUdp.begin(CMD_PORT)) {
        Serial.println("[comms] Failed to bind CMD port");
        return false;
    }
    if (!_telUdp.begin(0)) {  // ephemeral port for sending
        Serial.println("[comms] Failed to create telemetry socket");
        return false;
    }

    _lastCmdTime = millis();
    Serial.printf("[comms] Listening on UDP port %d\n", CMD_PORT);
    return true;
}

bool Comms::receiveCommand(Command &cmd) {
    int packetSize = _cmdUdp.parsePacket();
    if (packetSize < 5) {
        cmd.valid = false;
        return false;
    }

    uint8_t buf[8];
    int bytesRead = _cmdUdp.read(buf, sizeof(buf));

    if (bytesRead < 5) {
        cmd.valid = false;
        return false;
    }

    // Remember sender for telemetry replies
    _pcAddr = _cmdUdp.remoteIP();
    _pcPort = _cmdUdp.remotePort();
    _pcKnown = true;
    _lastCmdTime = millis();

    // Parse 5-byte base packet (big-endian)
    cmd.throttle  = (int16_t)((buf[0] << 8) | buf[1]);
    cmd.steering  = (int16_t)((buf[2] << 8) | buf[3]);
    cmd.buttons   = buf[4];

    // Extended 8-byte packet (backward compatible)
    if (bytesRead >= 8) {
        cmd.mode          = buf[5];
        cmd.heading_delta = (int16_t)((buf[6] << 8) | buf[7]);
    } else {
        cmd.mode          = MODE_DIRECT;
        cmd.heading_delta = 0;
    }

    cmd.valid = true;
    return true;
}

void Comms::sendTelemetry(const Telemetry &tel) {
    if (!_pcKnown) return;

    // 20-byte packet, little-endian floats + uint32
    uint8_t buf[20];
    memcpy(&buf[0],  &tel.heading,   4);
    memcpy(&buf[4],  &tel.gyro_z,    4);
    memcpy(&buf[8],  &tel.accel_x,   4);
    memcpy(&buf[12], &tel.accel_y,   4);
    memcpy(&buf[16], &tel.timestamp, 4);

    _telUdp.beginPacket(_pcAddr, TELEMETRY_PORT);
    _telUdp.write(buf, sizeof(buf));
    _telUdp.endPacket();
}

unsigned long Comms::timeSinceLastCmd() const {
    return millis() - _lastCmdTime;
}
