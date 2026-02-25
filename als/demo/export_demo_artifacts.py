"""
Generate canonical ALS demo artifacts for Three.js playback.

Usage:
    python3 -m als.demo.export_demo_artifacts
"""

from __future__ import annotations

import json
from pathlib import Path

from als.exporter import canonical_export_path, to_threejs_json, write_threejs_export
from als.fire import FireSpreadModel
from als.flood import FloodLoadModel


def write_demo_artifacts() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    fire_material_map = [
        [0.2, 0.4, 0.6, 0.4, 0.2],
        [0.3, 0.7, 0.9, 0.7, 0.3],
        [0.4, 0.8, 1.0, 0.8, 0.4],
        [0.3, 0.7, 0.9, 0.7, 0.3],
        [0.2, 0.4, 0.6, 0.4, 0.2],
    ]
    fire_frames = FireSpreadModel().run(
        material_map=fire_material_map,
        ignition_points=[(2, 2)],
        steps=48,
    )
    fire_payload = to_threejs_json(
        scenario_type="fire",
        frames=fire_frames,
        metadata={
            "demo": True,
            "model": "FireSpreadModel",
            "ignition_points": [(2, 2)],
        },
    )

    flood_elevation_grid = [
        [1.00, 0.95, 0.90, 0.88, 0.86],
        [0.98, 0.93, 0.88, 0.84, 0.82],
        [0.95, 0.90, 0.85, 0.80, 0.78],
        [0.93, 0.88, 0.83, 0.79, 0.75],
        [0.91, 0.86, 0.81, 0.77, 0.73],
    ]
    flood_frames = FloodLoadModel().run(
        elevation_grid=flood_elevation_grid,
        inflow_rate_m_s=0.05,
        structural_tolerance_pa=12000.0,
        steps=48,
    )
    flood_payload = to_threejs_json(
        scenario_type="flood",
        frames=flood_frames,
        metadata={
            "demo": True,
            "model": "FloodLoadModel",
            "inflow_rate_m_s": 0.05,
        },
    )

    fire_demo_path = output_dir / "als_demo_fire.json"
    flood_demo_path = output_dir / "als_demo_flood.json"
    fire_demo_path.write_text(json.dumps(fire_payload, indent=2), encoding="utf-8")
    flood_demo_path.write_text(json.dumps(flood_payload, indent=2), encoding="utf-8")

    canonical_fire_path = write_threejs_export(
        fire_payload,
        case_id="demo-case",
        scenario_type="fire",
        workspace_root=repo_root,
    )
    canonical_flood_path = write_threejs_export(
        flood_payload,
        case_id="demo-case",
        scenario_type="flood",
        workspace_root=repo_root,
    )

    return {
        "demo_fire_path": str(fire_demo_path),
        "demo_flood_path": str(flood_demo_path),
        "canonical_fire_path": str(canonical_fire_path),
        "canonical_flood_path": str(canonical_flood_path),
        "canonical_pattern": canonical_export_path("demo-case", "fire"),
    }


def main() -> None:
    paths = write_demo_artifacts()
    print("ALS demo artifacts generated:")
    for key, value in paths.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
