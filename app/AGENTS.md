# app/ — the browser frontend and its mirrored math

Static HTML, CSS and JavaScript. `app/js/math/` mirrors the Python math package module
for module.

## Rules

- **The mirror is the contract.** `app/js/math/free-energy.js` must match
  `src/active_inference/math/free_energy.py` numerically; a change to one needs the other
  in the same change. A silent divergence makes the browser demonstration wrong while the
  Python tests stay green.
- **Keep the module structure aligned.** `agent.js`, `environments.js`, `ui.js` and the
  `math/` files correspond to the Python subpackages; do not merge or rename one side
  alone.
- **Static and local.** No build step and no network calls; the page must run from local
  files served by `scripts/serve_app.py`.
- **Do not present the visualisation as empirical evidence.** It is an interactive
  demonstration of the model, not a measurement.
