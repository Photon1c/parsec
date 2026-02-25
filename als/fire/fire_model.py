"""
Deterministic fire spread model (v1).

Grid-based heat diffusion + material burn rate + structural degradation.
Designed for educational pre-visualization outputs.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class FireSpreadModel:
    def __init__(
        self,
        *,
        diffusion_coeff: float = 0.22,
        cooling_rate: float = 0.015,
        combustion_gain: float = 0.18,
        burnout_rate: float = 0.035,
        damage_gain: float = 0.04,
        time_step_s: float = 1.0,
    ):
        self.diffusion_coeff = diffusion_coeff
        self.cooling_rate = cooling_rate
        self.combustion_gain = combustion_gain
        self.burnout_rate = burnout_rate
        self.damage_gain = damage_gain
        self.time_step_s = time_step_s

    def run(
        self,
        material_map: list[list[float]],
        ignition_points: list[tuple[int, int]],
        steps: int = 30,
    ) -> list[dict[str, Any]]:
        """
        material_map values:
        - 0.0: non-flammable
        - 1.0: highly flammable
        """
        height = len(material_map)
        width = len(material_map[0]) if height else 0

        heat = [[0.0 for _ in range(width)] for _ in range(height)]
        damage = [[0.0 for _ in range(width)] for _ in range(height)]
        # Fuel depletes over time so heat can peak and decay.
        fuel = [[max(0.0, min(1.0, material_map[y][x])) for x in range(width)] for y in range(height)]

        for iy, ix in ignition_points:
            if 0 <= iy < height and 0 <= ix < width:
                heat[iy][ix] = 1.0

        frames: list[dict[str, Any]] = []
        for step in range(max(1, steps)):
            next_heat = deepcopy(heat)
            for y in range(height):
                for x in range(width):
                    local_heat = heat[y][x]
                    neighbors = self._neighbors(heat, x, y)
                    neighbor_avg = sum(neighbors) / len(neighbors) if neighbors else 0.0
                    diffusion = (neighbor_avg - local_heat) * self.diffusion_coeff
                    burn_rate = max(0.0, min(1.0, material_map[y][x])) * fuel[y][x]
                    combustion = burn_rate * local_heat * self.combustion_gain

                    updated = local_heat + diffusion + combustion - self.cooling_rate
                    next_heat[y][x] = max(0.0, min(1.0, updated))
                    fuel[y][x] = max(0.0, fuel[y][x] - local_heat * self.burnout_rate)

            heat = next_heat
            for y in range(height):
                for x in range(width):
                    damage[y][x] = max(0.0, min(1.0, damage[y][x] + heat[y][x] * self.damage_gain))

            frames.append(
                {
                    "t": round(step * self.time_step_s, 3),
                    "heat_grid": deepcopy(heat),
                    "damage_grid": deepcopy(damage),
                    "fuel_grid": deepcopy(fuel),
                }
            )
        return frames

    @staticmethod
    def _neighbors(grid: list[list[float]], x: int, y: int) -> list[float]:
        points = []
        h = len(grid)
        w = len(grid[0]) if h else 0
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                points.append(grid[ny][nx])
        return points
