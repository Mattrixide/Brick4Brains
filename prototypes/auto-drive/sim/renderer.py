"""Pygame renderer for the combat simulator. Separated from physics for headless testing."""
import math

import pygame

from .arena import SimArena


# Colors
BG_COLOR = (30, 30, 30)
GRID_COLOR = (0, 60, 0)
BORDER_COLOR = (0, 200, 0)
CROSSHAIR_COLOR = (0, 255, 0)
PIT_COLOR = (180, 0, 0)
PIT_TEXT_COLOR = (255, 80, 80)
BRICK_COLOR = (0, 180, 0)
ENEMY_COLOR = (200, 0, 0)
DEAD_COLOR = (100, 100, 100)
OUTLINE_COLOR = (255, 255, 255)
ARROW_COLOR = (255, 255, 255)


class SimRenderer:
    """Draws the arena, robots, and overlays to a pygame surface."""

    def __init__(self, arena: SimArena):
        self.arena = arena
        self._font = None

    def _get_font(self):
        if self._font is None:
            self._font = pygame.font.SysFont("consolas", 14)
        return self._font

    def cm_to_px(self, x, y, screen):
        """Convert world cm coordinates to screen pixel coordinates."""
        cfg = self.arena.cfg
        sw, sh = screen.get_size()

        # Center of arena bounding box
        corners = self.arena.arena_corners
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0

        # Scale and center on screen
        px = sw / 2.0 + (x - cx) * cfg.scale_px_per_cm
        py = sh / 2.0 - (y - cy) * cfg.scale_px_per_cm  # flip Y for screen coords
        return (int(px), int(py))

    def draw(self, screen):
        """Draw the full arena scene."""
        screen.fill(BG_COLOR)
        self._draw_grid(screen)
        self._draw_border(screen)
        self._draw_crosshair(screen)
        self._draw_pit(screen)
        self._draw_robot(screen, self.arena.brick, BRICK_COLOR)
        self._draw_robot(screen, self.arena.enemy, ENEMY_COLOR)

    def _draw_grid(self, screen):
        """Draw 30cm grid lines."""
        cfg = self.arena.cfg
        corners = self.arena.arena_corners
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        half_w = (max(xs) - min(xs)) / 2.0
        half_h = (max(ys) - min(ys)) / 2.0

        spacing = 30.0  # cm
        sw, sh = screen.get_size()

        # Vertical lines
        x = cx - (int(half_w / spacing) + 1) * spacing
        while x <= cx + half_w + spacing:
            px, _ = self.cm_to_px(x, 0, screen)
            if 0 <= px <= sw:
                pygame.draw.line(screen, GRID_COLOR, (px, 0), (px, sh), 1)
            x += spacing

        # Horizontal lines
        y = cy - (int(half_h / spacing) + 1) * spacing
        while y <= cy + half_h + spacing:
            _, py = self.cm_to_px(0, y, screen)
            if 0 <= py <= sh:
                pygame.draw.line(screen, GRID_COLOR, (0, py), (sw, py), 1)
            y += spacing

    def _draw_border(self, screen):
        """Draw arena border polygon."""
        pts = [self.cm_to_px(c[0], c[1], screen) for c in self.arena.arena_corners]
        if len(pts) >= 3:
            pygame.draw.polygon(screen, BORDER_COLOR, pts, 2)

    def _draw_crosshair(self, screen):
        """Draw center crosshair."""
        corners = self.arena.arena_corners
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0

        center = self.cm_to_px(cx, cy, screen)
        size = 10
        pygame.draw.line(screen, CROSSHAIR_COLOR,
                         (center[0] - size, center[1]), (center[0] + size, center[1]), 1)
        pygame.draw.line(screen, CROSSHAIR_COLOR,
                         (center[0], center[1] - size), (center[0], center[1] + size), 1)

    def _draw_pit(self, screen):
        """Draw pit area and label."""
        if self.arena.pit_center is None:
            return

        cx, cy = self.arena.pit_center
        r = self.arena.pit_radius

        # Draw pit as polygon (circle approximation)
        num_pts = 32
        pts = []
        for i in range(num_pts):
            angle = 2 * math.pi * i / num_pts
            px, py = self.cm_to_px(
                cx + r * math.cos(angle),
                cy + r * math.sin(angle),
                screen,
            )
            pts.append((px, py))
        pygame.draw.polygon(screen, PIT_COLOR, pts)

        # Label
        font = self._get_font()
        label = font.render("PIT", True, PIT_TEXT_COLOR)
        center = self.cm_to_px(cx, cy, screen)
        rect = label.get_rect(center=center)
        screen.blit(label, rect)

    def _draw_robot(self, screen, robot, color):
        """Draw a robot as a filled rectangle with heading arrow."""
        if robot.alive:
            fill = color
        else:
            fill = DEAD_COLOR

        corners = robot.get_corners_world()
        pts = [self.cm_to_px(c[0], c[1], screen) for c in corners]

        # Filled polygon
        pygame.draw.polygon(screen, fill, pts)
        # White outline
        pygame.draw.polygon(screen, OUTLINE_COLOR, pts, 1)

        # Heading arrow: center to midpoint of front edge
        cx, cy = robot.position
        center_px = self.cm_to_px(cx, cy, screen)

        # Front edge midpoint (average of front-right and front-left corners)
        fr, fl = corners[0], corners[1]
        front_mid = ((fr[0] + fl[0]) / 2.0, (fr[1] + fl[1]) / 2.0)
        front_px = self.cm_to_px(front_mid[0], front_mid[1], screen)

        pygame.draw.line(screen, ARROW_COLOR, center_px, front_px, 2)
