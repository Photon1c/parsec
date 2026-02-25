"""
Deterministic flood/load overstress model (v1).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class FloodLoadModel:
    def __init__(
        self,
        *,
        gravity: float = 9.81,
        water_density_kg_m3: float = 1000.0,
        time_step_s: float = 1.0,
    ):
        self.gravity = gravity
        self.water_density_kg_m3 = water_density_kg_m3
        self.time_step_s = time_step_s

    def run(
        self,
        elevation_grid: list[list[float]],
        inflow_rate_m_s: float = 0.08,
        structural_tolerance_pa: float = 12000.0,
        steps: int = 40,
    ) -> list[dict[str, Any]]:
        height = len(elevation_grid)
        width = len(elevation_grid[0]) if height else 0

        water_level = min(min(row) for row in elevation_grid) if height else 0.0
        frames: list[dict[str, Any]] = []

        for step in range(max(1, steps)):
            water_level += inflow_rate_m_s * self.time_step_s
            depth_grid = [[0.0 for _ in range(width)] for _ in range(height)]
            pressure_grid = [[0.0 for _ in range(width)] for _ in range(height)]
            stress_grid = [[0.0 for _ in range(width)] for _ in range(height)]
            overload_cells = 0

            for y in range(height):
                for x in range(width):
                    depth = max(0.0, water_level - elevation_grid[y][x])
                    pressure = self.water_density_kg_m3 * self.gravity * depth
                    stress_ratio = pressure / structural_tolerance_pa if structural_tolerance_pa > 0 else 0.0

                    depth_grid[y][x] = depth
                    pressure_grid[y][x] = pressure
                    stress_grid[y][x] = stress_ratio
                    if stress_ratio > 1.0:
                        overload_cells += 1

            frames.append(
                {
                    "t": round(step * self.time_step_s, 3),
                    "water_level_m": round(water_level, 4),
                    "depth_grid": deepcopy(depth_grid),
                    "pressure_grid_pa": deepcopy(pressure_grid),
                    "stress_ratio_grid": deepcopy(stress_grid),
                    "overload_cell_count": overload_cells,
                }
            )
        return frames
