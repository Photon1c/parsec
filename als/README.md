# ALS Mini README

Advanced Load Simulation (`/als`) is a deterministic simulation layer for PARSEC.

> **Important:** ALS outputs are approximate and for training/internal hypothesis development only.  
> **NOT evidentiary** and not for legal use.

## What is in `/als`

- `scene_geometry_agent.py` — normalize scene geometry (`source`, `confidence`, `UNKNOWN`)
- `hypothesis_agent.py` — generate competing hypotheses (`H1..H4`)
- `load_simulation_agent.py` — run per-hypothesis load/physics simulation
- `hypothesis_consistency_agent.py` — rank hypotheses by plausibility
- `core/simulation_engine.py` — deterministic rigid-body approximation engine
- `fire/fire_model.py` — fire spread starter model
- `flood/flood_model.py` — flood/load-overstress starter model
- `exporter/to_threejs_json.py` — JSON export for Three.js playback
- `demo/export_demo_artifacts.py` — creates demo artifacts

## Quick start

From repository root:

```bash
python3 als_demo_export.py
```

This writes:

- `output/als_demo_fire.json`
- `output/als_demo_flood.json`

And canonical exports:

- `als_exports/demo-case/fire.json`
- `als_exports/demo-case/flood.json`

## Canonical export path

Use this path convention for viewers (e.g., Tronverse/Three.js):

```text
/als_exports/<case_id>/<scenario_type>.json
```

Helper utilities:

- `canonical_export_path(case_id, scenario_type)`
- `write_threejs_export(payload, case_id=..., scenario_type=...)`

## Minimal Python usage

```python
from als.scene_geometry_agent import SceneGeometryAgent
from als.hypothesis_agent import HypothesisGeneratorAgent
from als.load_simulation_agent import LoadSimulationAgent
from als.hypothesis_consistency_agent import HypothesisConsistencyAgent

geometry_agent = SceneGeometryAgent()
scene_model = geometry_agent.default_scene_model(scenes=[])
sim_scene_model = geometry_agent.to_simulation_model(scene_model)

hypotheses = HypothesisGeneratorAgent().default_hypotheses()
load_agent = LoadSimulationAgent()
results = [load_agent.run(sim_scene_model, h).to_dict() for h in hypotheses]

ranking = HypothesisConsistencyAgent().rank_hypotheses(hypotheses, results, scene_model)
print(ranking["best_fit_hypothesis_id"])
```

## Tests

Run ALS sanity tests:

```bash
python3 -m unittest discover -s tests -p "test_als_models.py"
```
