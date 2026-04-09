"""Combat robot simulator — main entry point.
Run: python -m sim.run  (from prototypes/auto-drive/)
"""
import math
import sys
import os

# Ensure auto-drive is on the path for battle code imports (used by later tasks)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame

from sim.arena import SimArena
from sim.renderer import SimRenderer
from sim.config import SimConfig
from sim.bridge import SimBridge


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
    brick_bridge = SimBridge(arena.brick, cfg)
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
                elif event.key == pygame.K_r:
                    arena.reset()
                    brick_bridge.reset()
                    match_started = False
                    paused = True

        # --- Input + Physics (only when running) ---
        if not paused:
            keys = pygame.key.get_pressed()

            # Brick: AI or WASD
            if brick_ai:
                if not match_started:
                    brick_bridge.start_match(arena.enemy)
                    match_started = True
                brick_bridge.tick(1.0 / cfg.render_fps, arena.enemy)
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

            # Enemy: Arrow keys
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
            arena.enemy.apply_drive(enemy_throttle, enemy_steering, arena.cfg)

            arena.step()

        # --- Render ---
        renderer.draw(screen)

        # --- HUD ---
        # State label
        state_text = "PAUSED" if paused else "RUNNING"
        state_color = (255, 255, 0) if paused else (0, 255, 0)
        label = font_large.render(state_text, True, state_color)
        screen.blit(label, (10, 10))

        # AI state
        if brick_ai:
            ai_text = f"AI: {brick_bridge.state}"
            ai_label = font.render(ai_text, True, (0, 200, 255))
            screen.blit(ai_label, (10, 75))

        # Robot speeds
        bv = arena.brick.velocity
        brick_speed = math.sqrt(bv[0] ** 2 + bv[1] ** 2)
        ev = arena.enemy.velocity
        enemy_speed = math.sqrt(ev[0] ** 2 + ev[1] ** 2)

        brick_label = font.render(
            f"Brick: {brick_speed:.0f} cm/s  {'DEAD' if not arena.brick.alive else ''}",
            True, (0, 180, 0),
        )
        enemy_label = font.render(
            f"Enemy: {enemy_speed:.0f} cm/s  {'DEAD' if not arena.enemy.alive else ''}",
            True, (200, 0, 0),
        )
        screen.blit(brick_label, (10, 35))
        screen.blit(enemy_label, (10, 55))

        # Controls hint
        hint = font.render(
            "WASD=Brick  Arrows=Enemy  B=AI  Space=Pause  R=Reset  Esc=Quit",
            True, (140, 140, 140),
        )
        screen.blit(hint, (10, win_size - 25))

        pygame.display.flip()
        clock.tick(cfg.render_fps)

    pygame.quit()


if __name__ == "__main__":
    main()
