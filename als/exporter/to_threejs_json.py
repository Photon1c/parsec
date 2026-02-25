"""
Exports ALS simulation frames to a Three.js-friendly JSON schema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "als.threejs.v1"
ALS_WARNING = (
    "Simulation output is approximate and for educational/internal "
    "hypothesis development only. NOT evidentiary."
)
DEFAULT_EXPORT_ROOT = "als_exports"


def to_threejs_json(
    *,
    scenario_type: str,
    frames: list[dict[str, Any]],
    coordinate_system: str = "local_meters",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "scenario_type": scenario_type,
        "coordinate_system": coordinate_system,
        "metadata": metadata or {},
        "frame_count": len(frames),
        "frames": frames,
        "warning": ALS_WARNING,
    }


def canonical_export_path(
    case_id: str,
    scenario_type: str,
    *,
    base_dir: str = DEFAULT_EXPORT_ROOT,
) -> str:
    """
    Canonical path convention for Three.js loaders:
        /als_exports/<case_id>/<scenario_type>.json
    """
    safe_case = _safe_segment(case_id)
    safe_scenario = _safe_segment(scenario_type)
    return str(Path(base_dir) / safe_case / f"{safe_scenario}.json")


def write_threejs_export(
    payload: dict[str, Any],
    *,
    case_id: str,
    scenario_type: str,
    workspace_root: str | Path = ".",
    base_dir: str = DEFAULT_EXPORT_ROOT,
) -> Path:
    """Write a Three.js payload using the canonical ALS export path."""
    relative_path = canonical_export_path(case_id, scenario_type, base_dir=base_dir)
    target = Path(workspace_root) / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return target


def _safe_segment(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value).strip())
    text = text.strip("-")
    return text or "unknown"
