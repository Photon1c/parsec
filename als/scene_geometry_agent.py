"""
Scene Geometry Agent normalization layer.

This module focuses on producing a stable scene_model contract with
"source" and "confidence" metadata for inferred values.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


class SceneGeometryAgent:
    SYSTEM_PROMPT = """You are a Scene Geometry Agent for a reconstruction simulation pipeline.
Given incident transcript + scene breakdown, output normalized JSON with this exact top-level structure:
{
  "coordinate_system": "local_meters",
  "entities": {...},
  "environment": {...},
  "constraints": {...}
}

Requirements:
- Include source and confidence for inferred values:
  - source: "given" | "inferred" | "unknown"
  - confidence: number 0.0 to 1.0
- Unknown values must be explicit via "UNKNOWN" markers.
- Keep output physically plausible and conservative.
- Do not claim certainty where evidence is weak.
Return JSON only."""

    def parse_response(self, content: str, scenes: list[dict[str, Any]]) -> dict[str, Any]:
        raw = self._extract_json(content)
        if raw is None:
            return self.default_scene_model(scenes)
        return self.normalize_scene_model(raw, scenes)

    def default_scene_model(self, scenes: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "coordinate_system": "local_meters",
            "entities": {
                "vehicle_1": {
                    "type": "car",
                    "mass_kg": {"value": 1500, "source": "inferred", "confidence": 0.4},
                    "initial_position": {"value": [0.0, 0.0, 0.0], "source": "inferred", "confidence": 0.4},
                    "initial_velocity": {"value": [20.0, 0.0, 0.0], "source": "inferred", "confidence": 0.4},
                },
                "rider": {
                    "type": "person",
                    "mass_kg": {"value": 80, "source": "inferred", "confidence": 0.5},
                    "initial_position": {"value": [4.0, 1.0, 1.0], "source": "inferred", "confidence": 0.5},
                    "initial_velocity": {"value": [0.0, 0.0, 0.0], "source": "unknown", "confidence": 0.0},
                },
            },
            "environment": {
                "road_grade_deg": {"value": 0.0, "source": "inferred", "confidence": 0.3},
                "barrier_height_m": {"value": 1.2, "source": "inferred", "confidence": 0.4},
                "barrier_x_m": {"value": 12.0, "source": "inferred", "confidence": 0.3},
                "gravity": {"value": 9.81, "source": "given", "confidence": 1.0},
                "scene_count_hint": {"value": len(scenes), "source": "given", "confidence": 1.0},
            },
            "constraints": {
                "body_impact_point": {"value": ["UNKNOWN", "UNKNOWN", "UNKNOWN"], "source": "unknown", "confidence": 0.0},
                "estimated_time_of_fall_s": {"value": "UNKNOWN", "source": "unknown", "confidence": 0.0},
            },
        }

    def normalize_scene_model(
        self,
        raw: dict[str, Any],
        scenes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        model = deepcopy(self.default_scene_model(scenes))
        model["coordinate_system"] = raw.get("coordinate_system", "local_meters")

        model["entities"] = self._normalize_entity_block(raw.get("entities", {}), model["entities"])
        model["environment"] = self._normalize_metric_block(raw.get("environment", {}), model["environment"])
        model["constraints"] = self._normalize_metric_block(raw.get("constraints", {}), model["constraints"])

        return model

    def to_simulation_model(self, scene_model: dict[str, Any]) -> dict[str, Any]:
        """
        Flatten source-aware scene model into numeric-first values for simulator.
        Unknown values are replaced with conservative defaults.
        """
        sim_model = {
            "coordinate_system": scene_model.get("coordinate_system", "local_meters"),
            "entities": {},
            "environment": {},
            "constraints": {},
        }

        for name, entity in scene_model.get("entities", {}).items():
            sim_model["entities"][name] = {
                "type": entity.get("type", "UNKNOWN"),
                "mass_kg": self._unbox(entity.get("mass_kg"), 80.0),
                "initial_position": self._unbox(entity.get("initial_position"), [0.0, 0.0, 1.0]),
                "initial_velocity": self._unbox(entity.get("initial_velocity"), [0.0, 0.0, 0.0]),
            }

        for key, value in scene_model.get("environment", {}).items():
            sim_model["environment"][key] = self._unbox(value, None)
        for key, value in scene_model.get("constraints", {}).items():
            sim_model["constraints"][key] = self._unbox(value, None)

        # Fallback defaults required by simulator.
        sim_model["environment"].setdefault("gravity", 9.81)
        sim_model["environment"].setdefault("barrier_height_m", 1.2)
        sim_model["environment"].setdefault("barrier_x_m", 12.0)
        sim_model["environment"].setdefault("ground_z_m", 0.0)
        return sim_model

    def _normalize_entity_block(self, raw_entities: Any, defaults: dict[str, Any]) -> dict[str, Any]:
        entities = deepcopy(defaults)
        if not isinstance(raw_entities, dict):
            return entities

        for name, raw_entity in raw_entities.items():
            if not isinstance(raw_entity, dict):
                continue
            base = entities.get(name, {"type": "UNKNOWN"})
            entities[name] = {
                "type": raw_entity.get("type", base.get("type", "UNKNOWN")),
                "mass_kg": self._normalize_measure(raw_entity.get("mass_kg"), default=base.get("mass_kg")),
                "initial_position": self._normalize_measure(
                    raw_entity.get("initial_position"),
                    default=base.get("initial_position"),
                ),
                "initial_velocity": self._normalize_measure(
                    raw_entity.get("initial_velocity"),
                    default=base.get("initial_velocity"),
                ),
            }
        return entities

    def _normalize_metric_block(self, raw_block: Any, defaults: dict[str, Any]) -> dict[str, Any]:
        block = deepcopy(defaults)
        if not isinstance(raw_block, dict):
            return block
        for key, value in raw_block.items():
            block[key] = self._normalize_measure(value, default=block.get(key))
        return block

    def _normalize_measure(self, value: Any, default: Any) -> dict[str, Any]:
        base = self._default_measure(default)
        if isinstance(value, dict):
            normalized = {
                "value": value.get("value", base["value"]),
                "source": value.get("source", base["source"]),
                "confidence": value.get("confidence", base["confidence"]),
            }
            return self._safe_measure(normalized)
        if value is None:
            return base
        return self._safe_measure({"value": value, "source": "inferred", "confidence": 0.5})

    @staticmethod
    def _default_measure(default: Any) -> dict[str, Any]:
        if isinstance(default, dict) and {"value", "source", "confidence"}.issubset(default.keys()):
            return {
                "value": default.get("value"),
                "source": default.get("source", "unknown"),
                "confidence": default.get("confidence", 0.0),
            }
        return {"value": default, "source": "unknown", "confidence": 0.0}

    @staticmethod
    def _safe_measure(measure: dict[str, Any]) -> dict[str, Any]:
        source = measure.get("source", "unknown")
        if source not in {"given", "inferred", "unknown"}:
            source = "unknown"
        try:
            confidence = float(measure.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return {
            "value": measure.get("value", "UNKNOWN"),
            "source": source,
            "confidence": confidence,
        }

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any] | None:
        candidate = content.strip()
        if "```json" in candidate:
            candidate = candidate.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in candidate:
            candidate = candidate.split("```", 1)[1].split("```", 1)[0]
        try:
            parsed = json.loads(candidate.strip())
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _unbox(value: Any, default: Any) -> Any:
        raw = value.get("value") if isinstance(value, dict) else value
        if raw in (None, "UNKNOWN"):
            return default
        return raw
