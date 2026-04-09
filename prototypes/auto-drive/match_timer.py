"""Match and pin timers for combat state machine."""

import time


class MatchTimer:
    """Countdown timer for match duration with urgency scalar and match phase."""

    def __init__(
        self,
        duration_s: float = 180.0,
        urgency_ramp_s: float = 60.0,
        phase_start_s: float = 30.0,
        phase_final_s: float = 30.0,
    ):
        self._duration = duration_s
        self._urgency_ramp = urgency_ramp_s
        # Validate phase boundaries don't exceed match duration
        total_phases = phase_start_s + phase_final_s
        if total_phases >= duration_s:
            # Scale proportionally to fit
            scale = (duration_s * 0.8) / total_phases
            phase_start_s *= scale
            phase_final_s *= scale
        self._phase_start_s = phase_start_s
        self._phase_final_s = phase_final_s
        self._start: float | None = None

    def start(self) -> None:
        self._start = time.perf_counter()

    @property
    def is_running(self) -> bool:
        return self._start is not None

    @property
    def elapsed_s(self) -> float:
        if self._start is None:
            return 0.0
        return time.perf_counter() - self._start

    @property
    def remaining_s(self) -> float:
        if self._start is None:
            return self._duration
        return max(0.0, self._duration - (time.perf_counter() - self._start))

    @property
    def is_expired(self) -> bool:
        if self._start is None:
            return False
        return (time.perf_counter() - self._start) >= self._duration

    @property
    def urgency(self) -> float:
        """0.0 when plenty of time, ramps to 1.0 at match end."""
        remaining = self.remaining_s
        if remaining >= self._urgency_ramp:
            return 0.0
        if remaining <= 0.0:
            return 1.0
        return 1.0 - (remaining / self._urgency_ramp)

    @property
    def phase(self) -> str:
        """Match phase: 'start', 'mid', 'final', or 'post'."""
        if self.is_expired:
            return "post"
        elapsed = self.elapsed_s
        if elapsed < self._phase_start_s:
            return "start"
        if elapsed >= self._duration - self._phase_final_s:
            return "final"
        return "mid"

    @property
    def duration_s(self) -> float:
        return self._duration

    @duration_s.setter
    def duration_s(self, value: float) -> None:
        self._duration = max(1.0, value)

    def reset(self) -> None:
        self._start = None


class PinTimer:
    """Countdown timer for pin hold duration."""

    def __init__(self, max_duration_s: float = 5.0):
        self._duration = max(1.0, min(10.0, max_duration_s))
        self._start: float | None = None

    def start(self) -> None:
        self._start = time.perf_counter()

    @property
    def is_running(self) -> bool:
        return self._start is not None

    @property
    def elapsed_s(self) -> float:
        if self._start is None:
            return 0.0
        return time.perf_counter() - self._start

    @property
    def remaining_s(self) -> float:
        if self._start is None:
            return self._duration
        return max(0.0, self._duration - (time.perf_counter() - self._start))

    @property
    def is_expired(self) -> bool:
        if self._start is None:
            return False
        return (time.perf_counter() - self._start) >= self._duration

    @property
    def duration_s(self) -> float:
        return self._duration

    @duration_s.setter
    def duration_s(self, value: float) -> None:
        self._duration = max(1.0, min(10.0, value))

    def reset(self) -> None:
        self._start = None
