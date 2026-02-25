"""
Ranks hypothesis simulation results and prepares provenance-friendly summaries.
"""

from __future__ import annotations

from statistics import mean
from typing import Any


class HypothesisConsistencyAgent:
    def rank_hypotheses(
        self,
        hypotheses: list[dict[str, Any]],
        simulation_results: list[dict[str, Any]],
        scene_model: dict[str, Any],
    ) -> dict[str, Any]:
        by_id = {str(h.get("hypothesis_id")): h for h in hypotheses}
        ranked = []

        for result in simulation_results:
            hid = str(result.get("hypothesis_id", "UNKNOWN"))
            errors = result.get("constraint_errors", {})
            avg_error = mean([float(v) for v in errors.values()]) if errors else 0.0

            base_score = float(result.get("plausibility_score", 0.0))
            feasibility_boost = 0.08 if result.get("is_feasible") else -0.08
            confidence_hint = float(by_id.get(hid, {}).get("confidence", 0.4))

            score = max(0.0, min(1.0, base_score + feasibility_boost + (confidence_hint - 0.5) * 0.15))
            ranked.append(
                {
                    "id": hid,
                    "score": round(score, 3),
                    "description": self._description(by_id.get(hid, {}), result),
                    "is_feasible": bool(result.get("is_feasible")),
                    "average_constraint_error_m": round(avg_error, 3),
                }
            )

        ranked.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        best = ranked[0]["id"] if ranked else "UNKNOWN"
        return {
            "hypothesis_scores": ranked,
            "best_fit_hypothesis_id": best,
            "notes": "Scores are heuristic and for training/hypothesis only.",
            "simulation_not_evidence": (
                "Physical simulations are approximate and may not reflect real-world dynamics. "
                "These outputs are for educational and internal hypothesis development only, "
                "NOT for evidentiary or legal use."
            ),
            "scene_coordinate_system": scene_model.get("coordinate_system", "local_meters"),
        }

    @staticmethod
    def summarize_for_provenance(
        ranking: dict[str, Any],
        simulation_results: list[dict[str, Any]],
        simulation_engine: dict[str, Any],
    ) -> dict[str, Any]:
        errors = []
        for result in simulation_results:
            constraint_errors = result.get("constraint_errors", {})
            if constraint_errors:
                errors.append(mean(float(v) for v in constraint_errors.values()))

        return {
            "simulation_engine": simulation_engine,
            "hypotheses_considered": [item.get("id") for item in ranking.get("hypothesis_scores", [])],
            "best_fit_hypothesis_id": ranking.get("best_fit_hypothesis_id"),
            "average_constraint_error_m": round(mean(errors), 3) if errors else None,
            "constraint_errors": {
                str(result.get("hypothesis_id")): result.get("constraint_errors", {})
                for result in simulation_results
            },
            "simulation_warning": ranking.get("simulation_not_evidence"),
        }

    @staticmethod
    def _description(hypothesis: dict[str, Any], result: dict[str, Any]) -> str:
        label = hypothesis.get("label", "unnamed hypothesis")
        feasible = "feasible" if result.get("is_feasible") else "weak fit"
        return f"{label} ({feasible})"
