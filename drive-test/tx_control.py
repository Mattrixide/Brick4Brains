"""TX15 transmitter control via SBUS protocol over USB serial.

Sends SBUS packets to the TX15 configured in Master/SBUS trainer mode.
The TX15 forwards these channels to the receiver on the robot.

Setup:
  1. Connect TX15 via USB
  2. In EdgeTX: System -> USB -> Serial (not Joystick)
  3. In EdgeTX: Model -> Trainer -> Mode: Master/SBUS
  4. Note the COM port (Device Manager -> Ports)

Channel mapping:
  CH1 (index 0) = Steering (left/right)
  CH2 (index 1) = Throttle (forward/backward)

SBUS protocol:
  - 100000 baud, 8E2 (8 data bits, even parity, 2 stop bits)
  - 25-byte frame: [0x0F] [22 bytes channel data] [flags] [0x00]
  - 16 channels, 11 bits each (values 172-1811, center 992)
  - Frames sent every ~14ms (roughly 70Hz)
  - Note: SBUS is normally inverted serial, but USB serial handles
    this transparently -- no inversion needed on the PC side.
"""

import time
from threading import Thread, Lock, Event

import serial
import serial.tools.list_ports


# SBUS constants
SBUS_HEADER = 0x0F
SBUS_FOOTER = 0x00
SBUS_NUM_CHANNELS = 16
SBUS_FRAME_SIZE = 25

# SBUS channel value range (11-bit, same mapping as CRSF)
# 172 -> 1000us, 992 -> 1500us, 1811 -> 2000us
SBUS_CENTER = 992
SBUS_MIN = 172
SBUS_MAX = 1811

# Standard RC PWM range
PWM_CENTER = 1500
PWM_MIN = 1000
PWM_MAX = 2000

# SBUS serial config
SBUS_BAUDRATE = 100000


def _pack_sbus_channels(channels):
    """Pack 16 channels (11-bit each) into 22 bytes for SBUS frame.

    SBUS packs channels in little-endian bit order:
      byte0  = ch0[7:0]
      byte1  = ch1[4:0]:ch0[10:8]
      byte2  = ch2[1:0]:ch1[10:5]
      ... etc.
    """
    bits = 0
    for i in range(SBUS_NUM_CHANNELS):
        bits |= (int(channels[i]) & 0x7FF) << (i * 11)
    return bits.to_bytes(22, byteorder='little')


def pwm_to_sbus(pwm_us):
    """Convert PWM microseconds (1000-2000) to SBUS value (172-1811)."""
    pwm_us = max(PWM_MIN, min(PWM_MAX, pwm_us))
    return int((pwm_us - PWM_MIN) / (PWM_MAX - PWM_MIN) * (SBUS_MAX - SBUS_MIN) + SBUS_MIN)


def normalized_to_sbus(value):
    """Convert normalized value (-1.0 to 1.0) to SBUS channel value."""
    value = max(-1.0, min(1.0, value))
    pwm = PWM_CENTER + value * (PWM_MAX - PWM_CENTER)
    return pwm_to_sbus(int(pwm))


def list_serial_ports():
    """List available serial ports (helps find the TX15)."""
    ports = serial.tools.list_ports.comports()
    return [(p.device, p.description) for p in ports]


class TX15Controller:
    """Control robot via TX15 transmitter using SBUS over USB serial.

    Args:
        port: Serial port (e.g. 'COM3'). None for dry-run mode.
        send_rate_hz: How often to send SBUS frames (~70Hz is standard).
    """

    CH_STEER = 0    # Channel 1 = steering
    CH_THROTTLE = 1  # Channel 2 = throttle

    def __init__(self, port=None, send_rate_hz=70):
        self._port_name = port
        self._send_interval = 1.0 / send_rate_hz
        self._serial = None
        self._dry_run = port is None

        # 16 channels, all centered
        self._channels = [SBUS_CENTER] * SBUS_NUM_CHANNELS
        self._lock = Lock()
        self._stop_event = Event()
        self._send_thread = None

        # Track last commanded values for display
        self._steer_norm = 0.0
        self._throttle_norm = 0.0

    def connect(self):
        """Open serial connection and start sending SBUS frames."""
        if self._dry_run:
            print("[TX15] DRY RUN mode -- no serial connection")
        else:
            print(f"[TX15] Connecting to {self._port_name} (SBUS 100000 8E2)...")
            self._serial = serial.Serial(
                port=self._port_name,
                baudrate=SBUS_BAUDRATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_EVEN,
                stopbits=serial.STOPBITS_TWO,
                timeout=0.1,
            )
            time.sleep(0.1)
            print(f"[TX15] Connected to {self._port_name}")

        # Start background sender
        self._stop_event.clear()
        self._send_thread = Thread(target=self._send_loop, daemon=True)
        self._send_thread.start()

    def disconnect(self):
        """Stop sending and close serial."""
        self.stop()
        time.sleep(0.1)
        self._stop_event.set()
        if self._send_thread:
            self._send_thread.join(timeout=1.0)
        if self._serial and self._serial.is_open:
            self._serial.close()
            print("[TX15] Disconnected")

    def steer(self, value):
        """Set steering: -1.0 = full left, 0.0 = center, 1.0 = full right."""
        self._steer_norm = max(-1.0, min(1.0, value))
        with self._lock:
            self._channels[self.CH_STEER] = normalized_to_sbus(value)

    def throttle(self, value):
        """Set throttle: -1.0 = full reverse, 0.0 = stop, 1.0 = full forward."""
        self._throttle_norm = max(-1.0, min(1.0, value))
        with self._lock:
            self._channels[self.CH_THROTTLE] = normalized_to_sbus(value)

    def stop(self):
        """Center all channels (neutral/stop)."""
        self._steer_norm = 0.0
        self._throttle_norm = 0.0
        with self._lock:
            for i in range(SBUS_NUM_CHANNELS):
                self._channels[i] = SBUS_CENTER

    @property
    def steer_value(self):
        return self._steer_norm

    @property
    def throttle_value(self):
        return self._throttle_norm

    def _build_sbus_frame(self):
        """Build a 25-byte SBUS frame."""
        with self._lock:
            channel_data = _pack_sbus_channels(self._channels)

        # Flags byte: bit0=ch17, bit1=ch18, bit2=frame_lost, bit3=failsafe
        # All clear for normal operation
        flags = 0x00

        return bytes([SBUS_HEADER]) + channel_data + bytes([flags, SBUS_FOOTER])

    def _send_loop(self):
        """Background thread: send SBUS frames at ~70Hz."""
        while not self._stop_event.is_set():
            frame = self._build_sbus_frame()
            if self._serial and self._serial.is_open:
                try:
                    self._serial.write(frame)
                except serial.SerialException as e:
                    print(f"[TX15] Serial error: {e}")
                    break
            self._stop_event.wait(self._send_interval)

    def __repr__(self):
        mode = "DRY RUN" if self._dry_run else self._port_name
        return (f"TX15({mode}) steer={self._steer_norm:+.2f} "
                f"throttle={self._throttle_norm:+.2f}")
