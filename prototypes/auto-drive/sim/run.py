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
    brick_bridge = SimBridge(arena.brick, cfg, strategy_override="charge")
    logger = SimLogger(arena, brick_bridge)
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
                elif event.key == pygame.K_r:
                    if logging_on:
                        logger.stop()
                        logging_on = False
                    arena.reset()
                    brick_bridge.reset()
                    match_started = False
                    paused = True

        # --- Input + Physics (only when running) ---
        dt = 1.0 / cfg.render_fps
        if not paused:
            keys = pygame.key.get_pressed()

            # Brick: AI or WASD
            if brick_ai:
                if not match_started:
                    brick_bridge.start_match(arena.enemy)
                    match_started = True
                brick_bridge.tick(dt, arena.enemy)
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

            # Enemy: Arrow keys (only apply drive if keys pressed)
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

            arena.step()

            # Log frame
            if logging_on:
                logger.log_frame(dt)

        # --- Render ---
        renderer.draw(screen, bridge=brick_bridge if brick_ai else None)

        # --- HUD ---
        state_text = "PAUSED" if paused else "RUNNING"
        state_color = (255, 255, 0) if paused else (0, 255, 0)
        label = font_large.render(state_text, True, state_color)
        screen.blit(label, (10, 10))

        if brick_ai:
            ai_label = font.render(f"AI: {brick_bridge.state}", True, (0, 200, 255))
            screen.blit(ai_label, (10, 75))

        bv = arena.brick.velocity
        brick_speed = math.hypot(bv[0], bv[1])
        ev = arena.enemy.velocity
        enemy_speed = math.hypot(ev[0], ev[1])

        screen.blit(font.render(
            f"Brick: {brick_speed:.0f} cm/s  {'DEAD' if not arena.brick.alive else ''}",
            True, (0, 180, 0)), (10, 35))
        screen.blit(font.render(
            f"Enemy: {enemy_speed:.0f} cm/s  {'DEAD' if not arena.enemy.alive else ''}",
            True, (200, 0, 0)), (10, 55))

        # Log indicator
        if logging_on:
            log_label = font.render("REC", True, (255, 0, 0))
            screen.blit(log_label, (win_size - 40, 10))

        hint = font.render(
            "WASD=Brick  Arrows=Enemy  B=AI  L=Log  Space=Pause  R=Reset",
            True, (140, 140, 140))
        screen.blit(hint, (10, win_size - 25))

        pygame.display.flip()
        clock.tick(cfg.render_fps)

    if logging_on:
        logger.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
