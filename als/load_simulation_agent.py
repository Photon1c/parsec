"""
Advanced Load / Physics Simulation Agent for PARSEC.

The public interface is intentionally simple:
    result = LoadSimulationAgent().run(scene_model, hypothesis)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, radians, sin, sqrt
from typing import Any

from .core.simulation_engine import RigidBodyApproxEngine


@dataclass
class SimulationResult:
    hypothesis_id: str
    is_feasible: bool
    constraint_errors: dict[str, float]
    max_impact_force_n: float
    trajectory: list[dict[str, float]]
    plausibility_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LoadSimulationAgent:
    def __init__(self, time_step: float = 0.01, max_time: float = 3.0):
        self.time_step = time_step
        self.max_time = max_time
        self.engine = RigidBodyApproxEngine(time_step=time_step, max_time=max_time)

    @property
    def engine_metadata(self) -> dict[str, Any]:
        return self.engine.ENGINE_METADATA

    def run(self, scene_model: dict[str, Any], hypothesis: dict[str, Any]) -> SimulationResult:
        entities = scene_model.get("entities", {})
        environment = scene_model.get("environment", {})
        constraints = scene_model.get("constraints", {})

        rider = entities.get("rider", {})
        mass_kg = self._as_float(rider.get("mass_kg"), 80.0)
        initial_position = self._vector(rider.get("initial_position"), [0.0, 0.0, 1.0])
        base_velocity = self._vector(rider.get("initial_velocity"), [0.0, 0.0, 0.0])

        hypothesis_velocity = self._velocity_from_hypothesis(hypothesis)
        initial_velocity = [
            hypothesis_velocity[0] if hypothesis_velocity else base_velocity[0],
            hypothesis_velocity[1] if hypothesis_velocity else base_velocity[1],
            hypothesis_velocity[2] if hypothesis_velocity else base_velocity[2],
        ]

        gravity = self._as_float(environment.get("gravity"), 9.81)
        barrier_height = self._as_float(environment.get("barrier_height_m"), None)
        barrier_x = self._as_float(environment.get("barrier_x_m"), None)
        ground_z = self._as_float(environment.get("ground_z_m"), 0.0)

        trace = self.engine.simulate_projectile(
            mass_kg=mass_kg,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            gravity=gravity,
            barrier_x_m=barrier_x,
            barrier_height_m=barrier_height,
            barrier_restitution=0.25,
            ground_z_m=ground_z,
        )

        errors = self._compute_constraint_errors(trace, constraints, barrier_height=barrier_height)
        plausibility_score = self._plausibility(errors, trace)
        is_feasible = (
            errors.get("impact_point_error_m", 0.0) <= 6.0
            and errors.get("time_of_fall_error_s", 0.0) <= 1.5
            and errors.get("barrier_clearance_error_m", 0.0) <= 1.0
        )

        return SimulationResult(
            hypothesis_id=str(hypothesis.get("hypothesis_id", "UNKNOWN")),
            is_feasible=is_feasible,
            constraint_errors=errors,
            max_impact_force_n=float(trace.max_impact_force_n),
            trajectory=trace.trajectory,
            plausibility_score=plausibility_score,
        )

    def _velocity_from_hypothesis(self, hypothesis: dict[str, Any]) -> list[float] | None:
        params = hypothesis.get("parameters", {})
        events = hypothesis.get("events", [])

        rel_vel = self._as_float(params.get("relative_velocity_mps"), None)
        angle = self._as_float(params.get("impact_angle_deg"), 0.0)
        launch_elevation = self._as_float(params.get("launch_elevation_deg"), 18.0)

        if rel_vel is None:
            for event in events:
                event_vel = self._as_float(event.get("relative_velocity"), None)
                if event_vel is not None:
                    rel_vel = event_vel
                    break

        if rel_vel is None:
            return None

        angle_rad = radians(angle)
        elev_rad = radians(launch_elevation)
        horiz = rel_vel * cos(elev_rad)
        vx = horiz * cos(angle_rad)
        vy = horiz * sin(angle_rad)
        vz = rel_vel * sin(elev_rad)

        # If hypothesis explicitly includes barrier launch, enforce positive vertical component.
        if any("launch" in str(event.get("action", "")).lower() for event in events):
            vz = max(vz, 3.0)

        return [vx, vy, vz]

    def _compute_constraint_errors(
        self,
        trace: Any,
        constraints: dict[str, Any],
        *,
        barrier_height: float | None,
    ) -> dict[str, float]:
        errors: dict[str, float] = {}

        expected_impact = constraints.get("body_impact_point")
        if isinstance(expected_impact, list) and len(expected_impact) >= 3:
            ex, ey, ez = self._vector(expected_impact, [0.0, 0.0, 0.0])
            sx, sy, sz = trace.impact_point
            errors["impact_point_error_m"] = sqrt((sx - ex) ** 2 + (sy - ey) ** 2 + (sz - ez) ** 2)
        else:
            errors["impact_point_error_m"] = 4.0

        expected_fall_time = self._as_float(constraints.get("estimated_time_of_fall_s"), None)
        if expected_fall_time is not None and trace.ground_impact_time_s is not None:
            errors["time_of_fall_error_s"] = abs(trace.ground_impact_time_s - expected_fall_time)
        else:
            errors["time_of_fall_error_s"] = 0.75

        # Clearance error is 0 when no barrier is modeled.
        if barrier_height is None:
            errors["barrier_clearance_error_m"] = 0.0
        else:
            max_z = max((frame.get("z", 0.0) for frame in trace.trajectory), default=0.0)
            errors["barrier_clearance_error_m"] = max(0.0, barrier_height - max_z)

        return {k: float(v) for k, v in errors.items()}

    @staticmethod
    def _plausibility(errors: dict[str, float], trace: Any) -> float:
        weighted_error = (
            0.6 * min(errors.get("impact_point_error_m", 0.0) / 10.0, 1.0)
            + 0.25 * min(errors.get("time_of_fall_error_s", 0.0) / 2.0, 1.0)
            + 0.15 * min(errors.get("barrier_clearance_error_m", 0.0) / 2.0, 1.0)
        )
        force_penalty = min(trace.max_impact_force_n / 20000.0, 1.0) * 0.1
        score = 1.0 - weighted_error - force_penalty
        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def _as_float(value: Any, default: float | None) -> float | None:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _vector(raw: Any, default: list[float]) -> list[float]:
        if not isinstance(raw, list) or len(raw) < 3:
            return default.copy()
        values: list[float] = []
        for idx, fallback in enumerate(default):
            try:
                values.append(float(raw[idx]))
            except (TypeError, ValueError, IndexError):
                values.append(float(fallback))
        return values
