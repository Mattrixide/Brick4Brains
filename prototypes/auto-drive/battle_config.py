"""Battle configuration dataclass with JSON persistence."""

import json
import logging
from dataclasses import asdict, dataclass, field

log = logging.getLogger(__name__)

VALID_SIDES = ("front", "back", "left", "right")
VALID_STRATEGIES = ("charge", "pit", "evade")
VALID_OPENINGS = ("fast_pin", "center", "avoid", "charge", "pit")


@dataclass
class BattleConfig:
    # Match timing
    match_duration_s: float = 60.0
    pin_duration_s: float = 5.0
    reverse_duration_s: float = 2.0
    unstick_oscillate_s: float = 1.5
    urgency_ramp_start_s: float = 60.0

    # Strategy
    safe_side: str = "front"
    strategy: str = "charge"

    # Pit zone (arena coordinates in cm)
    pit_x_cm: float = 0.0
    pit_y_cm: float = 0.0
    pit_radius_cm: float = 20.0
    pit_danger_radius_cm: float = 40.0

    # Arena geometry
    arena_width_cm: float = 244.0
    arena_height_cm: float = 244.0
    wall_threshold_cm: float = 80.0

    # Combat thresholds
    charge_close_range_cm: float = 15.0
    pin_escape_range_cm: float = 25.0
    acquire_frames: int = 20
    lost_timeout_frames: int = 45

    # Opening strategy
    opening_strategy: str = "charge"

    # Match phases
    phase_start_s: float = 30.0
    phase_final_s: float = 30.0
    mid_aggression: float = 0.8
    final_aggression: float = 1.0

    # Push commit window
    push_commit_s: float = 1.0
    stall_speed_threshold: float = 8.0  # cm/s (accounts for vision noise)

    # Victory dance
    victory_dance_duration_s: float = 3.0

    def __post_init__(self):
        self.pin_duration_s = max(1.0, min(10.0, self.pin_duration_s))
        if self.safe_side not in VALID_SIDES:
            log.warning("Invalid safe_side %r, defaulting to 'front'", self.safe_side)
            self.safe_side = "front"
        if self.strategy not in VALID_STRATEGIES:
            log.warning("Invalid strategy %r, defaulting to 'charge'", self.strategy)
            self.strategy = "charge"
        if self.opening_strategy not in VALID_OPENINGS:
            log.warning("Invalid opening_strategy %r, defaulting to 'charge'", self.opening_strategy)
            self.opening_strategy = "charge"
        self.push_commit_s = max(0.1, min(3.0, self.push_commit_s))
        self.stall_speed_threshold = max(1.0, min(20.0, self.stall_speed_threshold))
        self.victory_dance_duration_s = max(1.0, min(10.0, self.victory_dance_duration_s))

    def save(self, path: str = "battle_config.json") -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        log.info("Battle config saved to %s", path)

    @classmethod
    def load(cls, path: str = "battle_config.json") -> "BattleConfig":
        try:
            with open(path) as f:
                data = json.load(f)
            config = cls(**{k: v for k, v in data.items()
                           if k in cls.__dataclass_fields__})
            log.info("Battle config loaded from %s", path)
            return config
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            log.warning("Could not load battle config from %s: %s — using defaults", path, e)
            return cls()

    def update(self, **kwargs) -> None:
        """Update fields from a dict, re-running validation."""
        for k, v in kwargs.items():
            if k in self.__dataclass_fields__:
                setattr(self, k, v)
        self.__post_init__()
