"""Optional pygame visualizer for debugging individual simulation runs."""

import math
from collections import deque

import pygame

from simulator.physics import Arena, RobotBody


# Color palette
BG_COLOR = (26, 26, 46)
WALL_COLOR = (200, 200, 200)
FLOOR_COLOR = (40, 40, 60)
PIT_COLOR = (180, 40, 40)
OUR_COLOR = (60, 120, 255)
ENEMY_COLOR = (255, 80, 80)
TRAIL_OUR_COLOR = (60, 120, 255, 80)
TRAIL_ENEMY_COLOR = (255, 80, 80, 80)
TEXT_COLOR = (220, 220, 220)
HUD_BG = (20, 20, 35, 200)

# State → color mapping for badge
STATE_COLORS = {
    "wait": (80, 80, 80),
    "goto_center": (100, 136, 68),
    "acquire": (255, 200, 50),
    "charge_pursue": (255, 100, 50),
    "charge_flank": (255, 150, 50),
    "charge_reorient": (200, 100, 50),
    "pin": (0, 200, 200),
    "pit_position": (50, 200, 50),
    "pit_push": (50, 255, 50),
    "pit_commit": (0, 255, 0),
    "pit_abort": (200, 200, 50),
    "evade_retreat": (180, 50, 200),
    "evade_reposition": (150, 100, 200),
    "wall_reverse": (255, 100, 50),
    "unstick": (255, 150, 0),
    "lost_target": (100, 100, 100),
    "lost_aruco": (150, 100, 50),
    "victory_dance": (255, 215, 0),
}


class Visualizer:
    """Pygame-based arena visualizer for debugging sim runs."""

    def __init__(self, arena: Arena, scale: float = 3.0, speed: float = 1.0):
        self.arena = arena
        self.scale = scale
        self.speed = max(0.1, speed)

        # Window size
        self.arena_w = int(arena.half_w * 2 * scale)
        self.arena_h = int(arena.half_h * 2 * scale)
        self.hud_height = 80
        self.width = self.arena_w
        self.height = self.arena_h + self.hud_height

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("B4B Combat Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 14)
        self.font_large = pygame.font.SysFont("consolas", 20)

        # Trails
        self._our_trail: deque[tuple[float, float]] = deque(maxlen=120)
        self._enemy_trail: deque[tuple[float, float]] = deque(maxlen=120)
        self._frame_count = 0

    def _world_to_screen(self, x_cm: float, y_cm: float) -> tuple[int, int]:
        """Convert world coordinates (cm, origin center) to screen pixels."""
        sx = int((x_cm + self.arena.half_w) * self.scale)
        sy = int((self.arena.half_h - y_cm) * self.scale)  # flip Y
        return (sx, sy)

    def draw_frame(self, our_body: RobotBody, enemy_body: RobotBody,
                   state: str, sim_time: float, remaining_s: float,
                   enemy_detected: bool, enemy_tracking: bool = True) -> None:
        """Draw one frame of the simulation."""
        self._frame_count += 1

        # Only draw every Nth frame based on speed
        draw_interval = max(1, int(4 / self.speed))
        if self._frame_count % draw_interval != 0:
            return

        # Record trails
        self._our_trail.append((float(our_body.pos[0]), float(our_body.pos[1])))
        self._enemy_trail.append((float(enemy_body.pos[0]), float(enemy_body.pos[1])))

        # Clear
        self.screen.fill(BG_COLOR)

        # Draw arena floor
        floor_rect = pygame.Rect(0, 0, self.arena_w, self.arena_h)
        pygame.draw.rect(self.screen, FLOOR_COLOR, floor_rect)

        # Draw arena walls
        pygame.draw.rect(self.screen, WALL_COLOR, floor_rect, 3)

        # Draw center cross
        cx, cy = self._world_to_screen(0, 0)
        pygame.draw.line(self.screen, (60, 60, 80), (cx - 10, cy), (cx + 10, cy), 1)
        pygame.draw.line(self.screen, (60, 60, 80), (cx, cy - 10), (cx, cy + 10), 1)

        # Draw square pit
        if self.arena.has_pit:
            # Lip outline (expanded rect)
            lip = self.arena.pit_lip_cm
            lx1, ly1 = self._world_to_screen(
                self.arena.pit_min[0] - lip, self.arena.pit_max[1] + lip)
            lx2, ly2 = self._world_to_screen(
                self.arena.pit_max[0] + lip, self.arena.pit_min[1] - lip)
            lip_rect = pygame.Rect(lx1, ly1, lx2 - lx1, ly2 - ly1)
            pygame.draw.rect(self.screen, (120, 80, 40), lip_rect, 3)

            # Pit hole (main rect)
            px1, py1 = self._world_to_screen(
                self.arena.pit_min[0], self.arena.pit_max[1])
            px2, py2 = self._world_to_screen(
                self.arena.pit_max[0], self.arena.pit_min[1])
            pit_rect = pygame.Rect(px1, py1, px2 - px1, py2 - py1)
            pygame.draw.rect(self.screen, PIT_COLOR, pit_rect)

            # Fall zone (darker center, shrunk by lip)
            shrink = self.arena.pit_lip_cm * 1.5
            fx1, fy1 = self._world_to_screen(
                self.arena.pit_min[0] + shrink, self.arena.pit_max[1] - shrink)
            fx2, fy2 = self._world_to_screen(
                self.arena.pit_max[0] - shrink, self.arena.pit_min[1] + shrink)
            fall_rect = pygame.Rect(fx1, fy1, fx2 - fx1, fy2 - fy1)
            pygame.draw.rect(self.screen, (80, 20, 20), fall_rect)

            # Border
            pygame.draw.rect(self.screen, (100, 20, 20), pit_rect, 2)

        # Draw trails
        for i, (tx, ty) in enumerate(self._our_trail):
            alpha = int(80 * (i + 1) / len(self._our_trail)) if self._our_trail else 80
            sx, sy = self._world_to_screen(tx, ty)
            pygame.draw.circle(self.screen, (60, 120, 255), (sx, sy), 2)

        for i, (tx, ty) in enumerate(self._enemy_trail):
            sx, sy = self._world_to_screen(tx, ty)
            pygame.draw.circle(self.screen, (255, 80, 80), (sx, sy), 2)

        # Draw robots
        self._draw_robot(our_body, OUR_COLOR, "US")
        self._draw_robot(enemy_body, ENEMY_COLOR, "THEM")

        # Draw state label above our robot
        sx, sy = self._world_to_screen(float(our_body.pos[0]), float(our_body.pos[1]))
        state_color = STATE_COLORS.get(state, (150, 150, 150))
        label = self.font.render(state.upper().replace("_", " "), True, state_color)
        self.screen.blit(label, (sx - label.get_width() // 2, sy - 30))

        # Detection indicator — only show when tracking is truly lost, not just a frame drop
        if not enemy_tracking:
            lost_label = self.font.render("TARGET LOST", True, (255, 50, 50))
            self.screen.blit(lost_label, (sx - lost_label.get_width() // 2, sy - 45))
        elif not enemy_detected:
            # Brief frame drop — show subtle indicator
            drop_label = self.font.render("~", True, (200, 200, 50))
            self.screen.blit(drop_label, (sx - 4, sy - 45))

        # Draw HUD
        self._draw_hud(state, sim_time, remaining_s, our_body, enemy_body,
                       enemy_detected, enemy_tracking)

        pygame.display.flip()

        # Throttle framerate
        target_fps = 60 * self.speed
        self.clock.tick(target_fps)

    def _draw_robot(self, body: RobotBody, color: tuple, label: str) -> None:
        """Draw a robot as an oriented rectangle with heading arrow."""
        sx, sy = self._world_to_screen(float(body.pos[0]), float(body.pos[1]))

        # Get corners in world coords, convert to screen
        corners = body.corners()
        screen_corners = [self._world_to_screen(float(c[0]), float(c[1])) for c in corners]

        # Fill rectangle
        pygame.draw.polygon(self.screen, color, screen_corners)
        # Outline
        pygame.draw.polygon(self.screen, (255, 255, 255), screen_corners, 2)

        # Front edge highlighted (first two corners = front-left, front-right)
        pygame.draw.line(self.screen, (255, 255, 100), screen_corners[0], screen_corners[1], 3)

        # Heading arrow from center
        arrow_len = body.length * 0.7 * self.scale
        ax = sx + int(arrow_len * math.cos(body.heading))
        ay = sy - int(arrow_len * math.sin(body.heading))
        pygame.draw.line(self.screen, (255, 255, 255), (sx, sy), (ax, ay), 2)

        # Label below robot
        r = int(body.radius * self.scale)
        text = self.font.render(label, True, TEXT_COLOR)
        self.screen.blit(text, (sx - text.get_width() // 2, sy + r + 4))

    def _draw_hud(self, state: str, sim_time: float, remaining_s: float,
                  our: RobotBody, enemy: RobotBody, detected: bool,
                  enemy_tracking: bool = True) -> None:
        """Draw the heads-up display below the arena."""
        hud_y = self.arena_h
        pygame.draw.rect(self.screen, (30, 30, 50),
                         (0, hud_y, self.width, self.hud_height))

        # Timer
        mins = int(remaining_s) // 60
        secs = int(remaining_s) % 60
        timer_text = f"{mins}:{secs:02d}"
        timer_surf = self.font_large.render(timer_text, True, (0, 200, 200))
        self.screen.blit(timer_surf, (10, hud_y + 8))

        # State
        state_color = STATE_COLORS.get(state, (150, 150, 150))
        state_surf = self.font.render(state.upper(), True, state_color)
        self.screen.blit(state_surf, (100, hud_y + 10))

        # Distance
        dist = math.hypot(our.pos[0] - enemy.pos[0], our.pos[1] - enemy.pos[1])
        dist_surf = self.font.render(f"Dist: {dist:.0f}cm", True, TEXT_COLOR)
        self.screen.blit(dist_surf, (100, hud_y + 30))

        # Detection status
        if detected:
            det_color, det_text = (50, 200, 50), "TRACKING"
        elif enemy_tracking:
            det_color, det_text = (200, 200, 50), "COASTING"
        else:
            det_color, det_text = (255, 50, 50), "LOST"
        det_surf = self.font.render(det_text, True, det_color)
        self.screen.blit(det_surf, (250, hud_y + 10))

        # Speeds
        our_spd = our.speed()
        enemy_spd = enemy.speed()
        spd_surf = self.font.render(
            f"Us: {our_spd:.0f} cm/s  Enemy: {enemy_spd:.0f} cm/s", True, TEXT_COLOR
        )
        self.screen.blit(spd_surf, (250, hud_y + 30))

        # Elapsed
        elapsed_surf = self.font.render(f"T={sim_time:.1f}s", True, (150, 150, 150))
        self.screen.blit(elapsed_surf, (10, hud_y + 35))

        # Urgency bar
        urgency = max(0, 1.0 - remaining_s / 180.0) if remaining_s < 180 else 0
        bar_x, bar_y = self.width - 170, hud_y + 10
        bar_w, bar_h = 150, 16
        pygame.draw.rect(self.screen, (60, 60, 80), (bar_x, bar_y, bar_w, bar_h))
        fill_w = int(bar_w * urgency)
        if urgency > 0.7:
            bar_color = (255, 50, 50)
        elif urgency > 0.3:
            bar_color = (255, 200, 50)
        else:
            bar_color = (0, 200, 200)
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, fill_w, bar_h))
        urg_label = self.font.render(f"Urgency {urgency:.0%}", True, TEXT_COLOR)
        self.screen.blit(urg_label, (bar_x, bar_y + 20))

    def handle_events(self) -> bool:
        """Process pygame events. Returns False if user quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False
        return True

    def close(self) -> None:
        pygame.quit()
