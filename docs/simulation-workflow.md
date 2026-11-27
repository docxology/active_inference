# Finite simulation

Use `SimulationEngine` and `ModelConfig` from
`active_inference.research.simulations` in your source checkout. Register a
named model, then call `run_simulation` with the desired state, observation,
and action counts.

```python
from active_inference.research.simulations import ModelConfig, SimulationEngine

engine = SimulationEngine({})
engine.register_model(ModelConfig(name="finite", time_horizon=8, time_step=0.5))
trace = engine.run_simulation(
    "finite", {"n_states": 3, "n_observations": 4, "n_actions": 2}
)
assert len(trace["state_beliefs"]) == 8
```

The native implementation seeds NumPy at the start of each simulation. It draws
categorical likelihood and transition arrays, samples observations from the
predictive distribution, conditions the belief vector and scores candidate
actions. The action scores are a simplified greedy teaching heuristic, with
transition predictions used for candidate scoring. The loop retains
its conditioned belief vector for the next observation.

Read `state_beliefs` as posterior probabilities. The `accuracy` field is the
largest posterior component, a confidence measure. `free_energy` is the
implementation's diagnostic proxy. The returned timestamps are fixed fixture
metadata. Compare numerical arrays across repeated runs to verify replay.

The [offline example](QUICK_START.md) records the configuration, observations,
actions, posterior trajectories and repeat checks in `teaching_fixture.json`.
