"""
Exports ALS simulation frames to a Three.js-friendly JSON schema.
"""

from __future__ import annotations

from typing import Any


def to_threejs_json(
    *,
    scenario_type: str,
    frames: list[dict[str, Any]],
    coordinate_system: str = "local_meters",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "als.threejs.v1",
        "scenario_type": scenario_type,
        "coordinate_system": coordinate_system,
        "metadata": metadata or {},
        "frame_count": len(frames),
        "frames": frames,
        "warning": (
            "Simulation output is approximate and for educational/internal "
            "hypothesis development only. NOT evidentiary."
        ),
    }
