"""
Core deterministic simulation primitives for ALS.

This module intentionally keeps the interface stable so the backing
simulation implementation can later be replaced with a higher-fidelity
engine (PyBullet, MuJoCo, etc.) without changing agent wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any


@dataclass
class SimulationTrace:
    """Deterministic kinematic trace for one simulated body."""

    trajectory: list[dict[str, float]]
    barrier_contact: bool
    barrier_contact_time_s: float | None
    ground_impact_time_s: float | None
    impact_point: list[float]
    max_impact_force_n: float
    max_speed_mps: float


class RigidBodyApproxEngine:
    """
    Lightweight deterministic rigid-body approximation.

    v0.1 limitations are explicitly surfaced in metadata so consumers
    (and provenance reports) can communicate assumptions.
    """

    ENGINE_METADATA = {
        "name": "parsic_loadsim_v0.1",
        "type": "rigid_body_approximation",
        "limitations": [
            "2D kinematics only",
            "No deformation modeling",
            "Assumes constant gravity and flat road",
        ],
    }

    def __init__(self, time_step: float = 0.01, max_time: float = 3.0):
        self.time_step = max(0.001, float(time_step))
        self.max_time = max(self.time_step, float(max_time))

    def simulate_projectile(
        self,
        *,
        mass_kg: float,
        initial_position: list[float],
        initial_velocity: list[float],
        gravity: float = 9.81,
        barrier_x_m: float | None = None,
        barrier_height_m: float | None = None,
        barrier_restitution: float = 0.25,
        ground_z_m: float = 0.0,
    ) -> SimulationTrace:
        """
        Simulate one body with simple Euler integration and barrier/ground checks.

        Coordinates:
        - x: forward direction
        - y: lateral direction
        - z: vertical direction
        """

        x, y, z = self._coerce_vector(initial_position, default=[0.0, 0.0, 1.0])
        vx, vy, vz = self._coerce_vector(initial_velocity, default=[0.0, 0.0, 0.0])
        dt = self.time_step

        trajectory: list[dict[str, float]] = []
        barrier_contact = False
        barrier_contact_time_s = None
        ground_impact_time_s = None
        max_impact_force_n = 0.0
        max_speed = 0.0

        t = 0.0
        while t <= self.max_time:
            speed = sqrt(vx * vx + vy * vy + vz * vz)
            max_speed = max(max_speed, speed)
            trajectory.append(
                {
                    "t": float(t),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "vx": float(vx),
                    "vy": float(vy),
                    "vz": float(vz),
                }
            )

            prev_x = x
            prev_z = z
            prev_vz = vz

            # Gravity-only acceleration in v0.1.
            vz -= gravity * dt

            x += vx * dt
            y += vy * dt
            z += vz * dt

            # Barrier check: if body crosses barrier plane below barrier top, bounce.
            if (
                barrier_x_m is not None
                and barrier_height_m is not None
                and not barrier_contact
                and self._crossed_plane(prev_x, x, barrier_x_m)
                and min(prev_z, z) <= barrier_height_m
            ):
                barrier_contact = True
                barrier_contact_time_s = t
                x = barrier_x_m
                # Reflect forward velocity with energy loss.
                vx = -abs(vx) * max(0.0, min(1.0, barrier_restitution))

            # Ground impact check.
            if z <= ground_z_m and prev_z > ground_z_m:
                z = ground_z_m
                ground_impact_time_s = t
                delta_v = abs(vz - prev_vz)
                max_impact_force_n = max(
                    max_impact_force_n,
                    (mass_kg * delta_v / dt) if dt > 0 else 0.0,
                )
                break

            t += dt

        impact_point = [float(x), float(y), float(z)]
        return SimulationTrace(
            trajectory=trajectory,
            barrier_contact=barrier_contact,
            barrier_contact_time_s=barrier_contact_time_s,
            ground_impact_time_s=ground_impact_time_s,
            impact_point=impact_point,
            max_impact_force_n=float(max_impact_force_n),
            max_speed_mps=float(max_speed),
        )

    @staticmethod
    def _crossed_plane(start: float, end: float, plane: float) -> bool:
        return (start <= plane <= end) or (end <= plane <= start)

    @staticmethod
    def _coerce_vector(raw: Any, default: list[float]) -> list[float]:
        if not isinstance(raw, list) or len(raw) < 3:
            return default.copy()
        vec: list[float] = []
        for idx, fallback in enumerate(default):
            try:
                vec.append(float(raw[idx]))
            except (TypeError, ValueError, IndexError):
                vec.append(float(fallback))
        return vec
