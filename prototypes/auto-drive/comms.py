"""UDP communication module for ESP32 robot motor control.

Sends 5-byte or 8-byte UDP packets to an ESP32 controller:
  Bytes 0-1: int16 big-endian throttle (-32767 to 32767, forward positive)
  Bytes 2-3: int16 big-endian steering (-32767 to 32767, right positive)
  Byte 4:    uint8 button bitmask
  --- Extended (8-byte) ---
  Byte 5:    uint8 mode (0=direct, 1=gyro-turn)
  Bytes 6-7: int16 big-endian heading delta (0.01 degree units, mode 1 only)
"""

import socket
import struct
import time
import subprocess
import re


MAX_INT16 = 32767

# Command modes
MODE_DIRECT = 0
MODE_GYRO_TURN = 1


class RobotComms:
    """UDP communication with ESP32 robot controller."""

    def __init__(self, host="esp32wifi.local", port=4210):
        self._host = host
        self._port = port
        self._sock = None
        self._addr = None
        self._dry_run = not host
        self._last_log_time = 0.0
        self.packets_sent = 0

    @property
    def connected(self):
        """True if the UDP socket exists and is ready to send."""
        return self._sock is not None

    def connect(self):
        """Resolve hostname and create UDP socket.

        On Windows, mDNS (.local) resolution can fail through normal
        getaddrinfo. Falls back to parsing a ping subprocess response
        to extract the resolved IP.
        """
        if self._dry_run:
            print("[comms] dry-run mode — no socket created")
            return

        try:
            ip = self._resolve(self._host)
        except ConnectionError as e:
            print(f"[comms] WARNING: {e}")
            self._dry_run = True
            print("[comms] Falling back to dry-run mode")
            return
        self._addr = (ip, self._port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[comms] connected to {self._host} ({ip}:{self._port})")

    def send(self, throttle_norm, steering_norm, buttons=0):
        """Send a motor command packet.

        Args:
            throttle_norm: float -1.0 to 1.0 (forward positive)
            steering_norm: float -1.0 to 1.0 (right positive)
            buttons: uint8 button bitmask (default 0)
        """
        throttle_norm = max(-1.0, min(1.0, throttle_norm))
        steering_norm = max(-1.0, min(1.0, steering_norm))

        throttle = int(throttle_norm * MAX_INT16)
        steering = int(steering_norm * MAX_INT16)
        buttons = buttons & 0xFF

        packet = struct.pack(">hhB", throttle, steering, buttons)

        if self._dry_run:
            now = time.monotonic()
            if now - self._last_log_time >= 0.25:  # 4 Hz
                print(f"[comms] dry-run  thr={throttle:+6d}  str={steering:+6d}  btn=0x{buttons:02X}")
                self._last_log_time = now
            self.packets_sent += 1
            return

        try:
            self._sock.sendto(packet, self._addr)
        except OSError:
            pass  # network unreachable — ignore
        self.packets_sent += 1

    def send_turn(self, heading_delta_deg):
        """Send a gyro-assisted turn command (Mode 1).

        Args:
            heading_delta_deg: turn angle in degrees (positive = right)
        """
        # Pack heading delta as int16 in 0.01 degree units
        delta_units = int(heading_delta_deg * 100)
        delta_units = max(-32767, min(32767, delta_units))

        packet = struct.pack(">hhBBh",
                             0,              # throttle (ignored in turn mode)
                             0,              # steering (ignored in turn mode)
                             0,              # buttons
                             MODE_GYRO_TURN, # mode
                             delta_units)    # heading delta

        if self._dry_run:
            print(f"[comms] dry-run  TURN delta={heading_delta_deg:+.1f}°")
            self.packets_sent += 1
            return

        try:
            self._sock.sendto(packet, self._addr)
        except OSError:
            pass
        self.packets_sent += 1

    def stop(self):
        """Send 5 zero packets to ensure motors stop."""
        for _ in range(5):
            self.send(0.0, 0.0, 0)

    def close(self):
        """Stop motors and close the socket."""
        if self._sock or self._dry_run:
            self.stop()
        if self._sock:
            self._sock.close()
            self._sock = None
            self._addr = None
            print("[comms] socket closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(hostname):
        """Resolve hostname to IP, with Windows mDNS ping fallback."""
        # Try standard resolution first
        try:
            return socket.gethostbyname(hostname)
        except socket.gaierror:
            pass

        # Windows mDNS fallback: ping -n 1 hostname, parse the IP
        if hostname.endswith(".local"):
            try:
                result = subprocess.run(
                    ["ping", "-n", "1", "-w", "2000", hostname],
                    capture_output=True, text=True, timeout=5,
                )
                match = re.search(r"\[(\d+\.\d+\.\d+\.\d+)\]", result.stdout)
                if match:
                    return match.group(1)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        raise ConnectionError(f"Cannot resolve {hostname}")
