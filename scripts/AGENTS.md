# scripts/ — figure, simulation and server entry points

Three thin scripts: `generate_figures.py`, `run_simulation.py` and `serve_app.py`.

## Rules

- **Thin only.** Core math and agent behaviour live in `src/active_inference/`; a script
  reads inputs, calls the package and writes outputs.
- **The server binds locally.** `serve_app.py` exists to preview the frontend on this
  machine; it must not expose a public port or add authentication-free remote access.
- **Figures are regenerated from real runs**, never hand-edited to match prose.
- Keep execution deterministic: pass an explicit seed to a simulation.
- No network calls or credentials beyond what serving local files needs.
