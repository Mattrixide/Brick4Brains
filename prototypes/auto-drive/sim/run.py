"""Combat robot simulator — main entry point.
Run: python -m sim.run  (from prototypes/auto-drive/)
"""
import json
import math
import sys
import os
import time

# Ensure auto-drive is on the path for battle code imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame

from sim.arena import SimArena
from sim.renderer import SimRenderer
from sim.config import SimConfig
from sim.bridge import SimBridge
from sim.enemy_ai import EnemyController, MODES

AUTO_DRIVE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(AUTO_DRIVE_DIR, "logs")


class SimLogger:
    """Writes JSONL frame logs compatible with the replay viewer."""

    def __init__(self, arena, bridge):
        self.arena = arena
        self.bridge = bridge
        self.file = None
        self.frame_count = 0
        self.start_time = None
        self.log_path = None

    def start(self):
        os.makedirs(LOGS_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(LOGS_DIR, f"sim_{ts}.jsonl")
        self.file = open(self.log_path, "w")
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self._write_arena_meta()
        print(f"Logging to {self.log_path}")

    def _write_arena_meta(self):
        meta_path = self.log_path.replace(".jsonl", "_arena.json")
        arena = self.arena
        meta = {"arena_width_cm": 244.0, "arena_height_cm": 244.0}
        if arena.floor_cal:
            if "corners_ft" in arena.floor_cal:
                meta["corners_cm"] = arena.floor_cal["corners_ft"]
            if "inv_homography" in arena.floor_cal:
                meta["inv_homography"] = arena.floor_cal["inv_homography"]
            if "homography" in arena.floor_cal:
                meta["homography"] = arena.floor_cal["homography"]
            rgb = arena.floor_cal.get("rgb_size", [1280, 800])
            meta["frame_w"] = rgb[0]
            meta["frame_h"] = rgb[1]
            if "origin_x" in arena.floor_cal:
                meta["origin_x"] = arena.floor_cal["origin_x"]
                meta["origin_y"] = arena.floor_cal["origin_y"]
            if "px_per_cm" in arena.floor_cal:
                meta["px_per_cm"] = arena.floor_cal["px_per_cm"]
        if arena.pit_center:
            meta["pit_x_cm"] = arena.pit_center[0]
            meta["pit_y_cm"] = arena.pit_center[1]
            meta["pit_radius_cm"] = arena.pit_radius
            meta["pit_danger_radius_cm"] = arena.pit_radius + 15
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def log_frame(self, dt):
        if not self.file:
            return
        a = self.arena
        b = a.brick
        e = a.enemy
        bx, by = b.position
        bvx, bvy = b.velocity
        ex, ey = e.position
        evx, evy = e.velocity
        dist = math.hypot(ex - bx, ey - by) if e.alive else 999.0
        edge = max(0, dist - b.depth / 2 - e.depth / 2)

        br = self.bridge
        output = br.last_output
        thr = output.throttle if output.target_omega_dps is None else output.target_speed
        steer = output.steering
        bs = br.state
        mr = br.match_timer.remaining_s if br.match_timer.is_running else None
        urg = br.match_timer.urgency if br.match_timer.is_running else None
        mp = br.match_timer.phase if br.match_timer.is_running else None

        ab = [[round(c[0], 1), round(c[1], 1)] for c in b.get_corners_world()]
        fp = ab

        rec = {
            "f": self.frame_count,
            "t": round(time.perf_counter() - self.start_time, 4),
            "mode": "battle" if br.match_timer.is_running else "ready",
            "bs": bs,
            "mp": mp,
            "ox": round(bx, 1), "oy": round(by, 1),
            "oh": round(b.heading_rad, 3),
            "od": b.alive,
            "ovx": round(bvx, 1), "ovy": round(bvy, 1),
            "ex": round(ex, 1) if e.alive else None,
            "ey": round(ey, 1) if e.alive else None,
            "eh": round(e.heading_rad, 3) if e.alive else None,
            "evx": round(evx, 1) if e.alive else None,
            "evy": round(evy, 1) if e.alive else None,
            "ed": e.alive, "et": e.alive,
            "edx": round(ex, 1) if e.alive else None,
            "edy": round(ey, 1) if e.alive else None,
            "dist": round(edge, 1),
            "thr": round(thr, 3), "str": round(steer, 3),
            "mr": round(mr, 1) if mr is not None else None,
            "urg": round(urg, 3) if urg is not None else None,
            "fps": round(1 / dt, 1) if dt > 0 else 60.0,
            "ab": ab, "fp": fp,
            "ehm": "velocity", "ehc": 1.0,
            "ax": None, "ay": None,
        }
        self.file.write(json.dumps(rec) + "\n")
        self.frame_count += 1

    def stop(self):
        if self.file:
            self.file.close()
            self.file = None
            print(f"Log closed ({self.frame_count} frames) -> {self.log_path}")


def main():
    pygame.init()

    cfg = SimConfig.load()
    win_size = int(cfg.arena_cm * cfg.scale_px_per_cm + 80)
    screen = pygame.display.set_mode((win_size, win_size))
    pygame.display.set_caption("B4B Combat Simulator")
    clock = pygame.time.Clock()

    arena = SimArena()
    renderer = SimRenderer(arena)

    paused = True
    brick_ai = False
    match_started = False
    logging_on = False
    strategies = ["charge", "pit"]
    strategy_idx = 0
    SPEEDS = [0.5, 1.0, 2.0, 4.0]
    speed_idx = 1
    brick_bridge = SimBridge(arena.brick, cfg, strategy_override=strategies[strategy_idx])
    logger = SimLogger(arena, brick_bridge)
    enemy_ctrl = EnemyController()
    font_small = pygame.font.SysFont("consolas", 12)
    font = pygame.font.SysFont("consolas", 14)
    font_large = pygame.font.SysFont("consolas", 18)

    running = True
    while running:
        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_b:
                    brick_ai = not brick_ai
                elif event.key == pygame.K_l:
                    if logging_on:
                        logger.stop()
                        logging_on = False
                    else:
                        logger.start()
                        logging_on = True
                elif event.key == pygame.K_t:
                    strategy_idx = (strategy_idx + 1) % len(strategies)
                    brick_bridge = SimBridge(arena.brick, cfg,
                                            strategy_override=strategies[strategy_idx])
                    logger = SimLogger(arena, brick_bridge)
                    match_started = False
                    print(f"Strategy: {strategies[strategy_idx].upper()}")
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3,
                                    pygame.K_4, pygame.K_5, pygame.K_6):
                    mode_idx = event.key - pygame.K_1
                    enemy_ctrl.set_mode(MODES[mode_idx])
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    speed_idx = min(len(SPEEDS) - 1, speed_idx + 1)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed_idx = max(0, speed_idx - 1)
                elif event.key == pygame.K_r:
                    if logging_on:
                        logger.stop()
                        logging_on = False
                    arena.reset()
                    brick_bridge.reset()
                    enemy_ctrl.reset()
                    match_started = False
                    paused = True

        # --- Input + Physics (only when running) ---
        dt = 1.0 / cfg.render_fps
        sim_speed = SPEEDS[speed_idx]
        effective_dt = dt * sim_speed
        if not paused:
            keys = pygame.key.get_pressed()

            # Brick: AI or WASD
            if brick_ai:
                if not match_started:
                    brick_bridge.start_match(arena.enemy)
                    match_started = True
                brick_bridge.tick(effective_dt, arena.enemy)
            else:
                brick_throttle = 0.0
                brick_steering = 0.0
                if keys[pygame.K_w]:
                    brick_throttle = 1.0
                elif keys[pygame.K_s]:
                    brick_throttle = -1.0
                if keys[pygame.K_a]:
                    brick_steering = 1.0
                elif keys[pygame.K_d]:
                    brick_steering = -1.0
                arena.brick.apply_drive(brick_throttle, brick_steering, arena.cfg)

            # Enemy: freeze when match is over, otherwise drive
            match_over = brick_ai and match_started and brick_bridge.match_timer.is_expired
            if match_over:
                pass  # enemy stops — match is over
            elif enemy_ctrl.mode == "manual":
                enemy_throttle = 0.0
                enemy_steering = 0.0
                if keys[pygame.K_UP]:
                    enemy_throttle = 1.0
                elif keys[pygame.K_DOWN]:
                    enemy_throttle = -1.0
                if keys[pygame.K_LEFT]:
                    enemy_steering = 1.0
                elif keys[pygame.K_RIGHT]:
                    enemy_steering = -1.0
                if abs(enemy_throttle) > 0.01 or abs(enemy_steering) > 0.01:
                    arena.enemy.apply_drive(enemy_throttle, enemy_steering, arena.cfg)
            else:
                result = enemy_ctrl.get_drive(arena.enemy, arena.brick, effective_dt, arena.cfg)
                if result is not None:
                    arena.enemy.apply_drive(result[0], result[1], arena.cfg)

            arena.step(effective_dt)

            # Log frame
            if logging_on:
                logger.log_frame(effective_dt)

        # --- Render ---
        renderer.draw(screen, bridge=brick_bridge if brick_ai else None)

        # --- HUD ---
        y_hud = 10

        # State + speed indicator on same line
        state_text = "PAUSED" if paused else "RUNNING"
        state_color = (255, 255, 0) if paused else (0, 255, 0)
        label = font_large.render(state_text, True, state_color)
        screen.blit(label, (10, y_hud))
        if sim_speed != 1.0:
            spd_label = font_large.render(f" {sim_speed}x", True, (200, 200, 0))
            screen.blit(spd_label, (10 + label.get_width(), y_hud))
        y_hud += 24

        # Robot speeds
        bv = arena.brick.velocity
        brick_speed = math.hypot(bv[0], bv[1])
        ev = arena.enemy.velocity
        enemy_speed = math.hypot(ev[0], ev[1])

        brick_status = "DEAD" if not arena.brick.alive else f"{brick_speed:.0f} cm/s"
        enemy_status = "DEAD" if not arena.enemy.alive else f"{enemy_speed:.0f} cm/s"
        screen.blit(font.render(f"Brick: {brick_status}", True, (0, 180, 0)), (10, y_hud))
        y_hud += 18
        screen.blit(font.render(f"Enemy: {enemy_status}", True, (200, 0, 0)), (10, y_hud))
        y_hud += 22

        # AI / enemy mode info
        if brick_ai:
            ai_label = font.render(f"AI: {brick_bridge.state}  [{strategies[strategy_idx].upper()}]", True, (0, 200, 255))
            screen.blit(ai_label, (10, y_hud))
            y_hud += 18
        if enemy_ctrl.mode != "manual":
            enemy_mode_label = font.render(f"Enemy: {enemy_ctrl.mode.upper()}", True, (255, 160, 0))
            screen.blit(enemy_mode_label, (10, y_hud))

        # --- Bottom bar ---
        bar_y = win_size - 22

        # Controls hint (left)
        hint = font_small.render(
            "WASD/Arrows=Drive  B=AI  T=Strat  1-6=Enemy  +/-=Speed  L=Log  Space=Pause  R=Reset",
            True, (120, 120, 120))
        screen.blit(hint, (10, bar_y))

        # Right-aligned indicators
        right_x = win_size - 10
        if logging_on:
            rec_label = font.render("REC", True, (255, 0, 0))
            right_x -= rec_label.get_width()
            screen.blit(rec_label, (right_x, bar_y - 2))
            right_x -= 10
        if sim_speed != 1.0:
            spd = font.render(f"{sim_speed}x", True, (200, 200, 0))
            right_x -= spd.get_width()
            screen.blit(spd, (right_x, bar_y - 2))

        pygame.display.flip()
        clock.tick(cfg.render_fps)

    if logging_on:
        logger.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
