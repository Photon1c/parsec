"""
Hypothesis generation helpers for ALS.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


class HypothesisGeneratorAgent:
    SYSTEM_PROMPT = """You are a hypothesis generation agent for incident reconstruction simulation.
Generate 2-4 competing hypotheses from transcript + scene model.

Each hypothesis must be JSON object with:
{
  "hypothesis_id": "H1",
  "label": "short label",
  "events": [
    {"t": 0.0, "action": "...", "relative_velocity": 0.0}
  ],
  "parameters": {
    "impact_angle_deg": 0,
    "road_friction_coeff": 0.8
  },
  "source": "inferred",
  "confidence": 0.0-1.0
}

Rules:
- Ensure hypotheses are materially different.
- Keep conservative and uncertainty-aware.
- Unknowns must be explicit.
- Return JSON only, either an array or {"hypotheses": [...]}."""

    def parse_response(self, content: str) -> list[dict[str, Any]]:
        parsed = self._extract_json(content)
        hypotheses = self._extract_list(parsed) if parsed is not None else []
        normalized = [self._normalize_hypothesis(h, idx + 1) for idx, h in enumerate(hypotheses)]
        if len(normalized) < 2:
            return self.default_hypotheses()
        return normalized[:4]

    def default_hypotheses(self) -> list[dict[str, Any]]:
        return [
            {
                "hypothesis_id": "H1",
                "label": "self-initiated jump",
                "events": [
                    {"t": 0.0, "action": "rider_changes_direction", "relative_velocity": 2.5},
                    {"t": 0.2, "action": "rider_launches_over_barrier"},
                    {"t": 1.4, "action": "rider_impacts_ground"},
                ],
                "parameters": {
                    "impact_angle_deg": 5.0,
                    "relative_velocity_mps": 6.0,
                    "road_friction_coeff": 0.8,
                    "launch_elevation_deg": 26.0,
                },
                "source": "inferred",
                "confidence": 0.22,
            },
            {
                "hypothesis_id": "H2",
                "label": "rear impact ejection",
                "events": [
                    {"t": 0.0, "action": "vehicle_2_collides_with_rider", "relative_velocity": 15.0},
                    {"t": 0.1, "action": "rider_launches_over_barrier"},
                    {"t": 1.3, "action": "rider_impacts_ground"},
                ],
                "parameters": {
                    "impact_angle_deg": 10.0,
                    "relative_velocity_mps": 15.0,
                    "road_friction_coeff": 0.75,
                    "launch_elevation_deg": 18.0,
                },
                "source": "inferred",
                "confidence": 0.5,
            },
            {
                "hypothesis_id": "H3",
                "label": "side swipe and loss of control",
                "events": [
                    {"t": 0.0, "action": "vehicle_2_side_swipes_rider", "relative_velocity": 9.0},
                    {"t": 0.4, "action": "rider_loses_control"},
                    {"t": 1.7, "action": "rider_impacts_ground"},
                ],
                "parameters": {
                    "impact_angle_deg": 35.0,
                    "relative_velocity_mps": 9.0,
                    "road_friction_coeff": 0.7,
                    "launch_elevation_deg": 12.0,
                },
                "source": "inferred",
                "confidence": 0.38,
            },
        ]

    def _normalize_hypothesis(self, raw: dict[str, Any], ordinal: int) -> dict[str, Any]:
        hypothesis = deepcopy(raw if isinstance(raw, dict) else {})
        hypothesis_id = str(hypothesis.get("hypothesis_id", f"H{ordinal}")).strip() or f"H{ordinal}"
        label = str(hypothesis.get("label", "unnamed hypothesis")).strip()
        events = hypothesis.get("events", [])
        if not isinstance(events, list):
            events = []

        norm_events = []
        for event in events:
            if not isinstance(event, dict):
                continue
            norm_events.append(
                {
                    "t": self._safe_float(event.get("t"), 0.0),
                    "action": str(event.get("action", "UNKNOWN")),
                    "relative_velocity": self._safe_float(event.get("relative_velocity"), 0.0),
                }
            )
        if not norm_events:
            norm_events = [{"t": 0.0, "action": "UNKNOWN", "relative_velocity": 0.0}]

        params = hypothesis.get("parameters", {})
        if not isinstance(params, dict):
            params = {}

        normalized = {
            "hypothesis_id": hypothesis_id,
            "label": label,
            "events": norm_events,
            "parameters": {
                "impact_angle_deg": self._safe_float(params.get("impact_angle_deg"), 0.0),
                "road_friction_coeff": self._safe_float(params.get("road_friction_coeff"), 0.8),
                "relative_velocity_mps": self._safe_float(params.get("relative_velocity_mps"), None),
                "launch_elevation_deg": self._safe_float(params.get("launch_elevation_deg"), 18.0),
            },
            "source": str(hypothesis.get("source", "inferred")),
            "confidence": self._safe_float(hypothesis.get("confidence"), 0.4),
        }
        normalized["confidence"] = max(0.0, min(1.0, normalized["confidence"]))
        return normalized

    @staticmethod
    def _safe_float(value: Any, default: float | None) -> float | None:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any] | list[Any] | None:
        candidate = content.strip()
        if "```json" in candidate:
            candidate = candidate.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in candidate:
            candidate = candidate.split("```", 1)[1].split("```", 1)[0]
        try:
            return json.loads(candidate.strip())
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_list(parsed: Any) -> list[dict[str, Any]]:
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            candidates = parsed.get("hypotheses", [])
            if isinstance(candidates, list):
                return [item for item in candidates if isinstance(item, dict)]
        return []
