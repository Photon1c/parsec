import unittest

from als.exporter import ALS_WARNING, SCHEMA_VERSION, canonical_export_path, to_threejs_json
from als.fire import FireSpreadModel
from als.flood import FloodLoadModel


class AlsModelTests(unittest.TestCase):
    def test_flood_uniform_elevation_creates_uniform_stress(self):
        model = FloodLoadModel(time_step_s=1.0)
        elevation_grid = [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5], [1.5, 1.5, 1.5]]
        frames = model.run(
            elevation_grid=elevation_grid,
            inflow_rate_m_s=0.1,
            structural_tolerance_pa=10000.0,
            steps=3,
        )
        self.assertEqual(len(frames), 3)
        for frame in frames:
            values = [cell for row in frame["stress_ratio_grid"] for cell in row]
            first = values[0]
            for value in values[1:]:
                self.assertAlmostEqual(value, first, places=9)

    def test_fire_single_ignition_grows_then_decays(self):
        model = FireSpreadModel(time_step_s=1.0, burnout_rate=0.05)
        material_map = [[1.0 for _ in range(5)] for _ in range(5)]
        frames = model.run(material_map=material_map, ignition_points=[(2, 2)], steps=80)

        total_heat = [sum(sum(row) for row in frame["heat_grid"]) for frame in frames]
        peak_idx = total_heat.index(max(total_heat))

        self.assertGreater(peak_idx, 0)
        self.assertGreater(total_heat[peak_idx], total_heat[0])
        self.assertLess(total_heat[-1], total_heat[peak_idx])

    def test_threejs_schema_and_warning(self):
        frames = [{"t": 0.0, "heat_grid": [[1.0]]}]
        payload = to_threejs_json(scenario_type="fire", frames=frames)

        self.assertEqual(payload["schema"], SCHEMA_VERSION)
        self.assertEqual(payload["frame_count"], len(frames))
        self.assertEqual(payload["warning"], ALS_WARNING)
        self.assertEqual(
            canonical_export_path("Case 123", "fire spread"),
            "als_exports/Case-123/fire-spread.json",
        )


if __name__ == "__main__":
    unittest.main()
