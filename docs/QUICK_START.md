# Quick start

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). Run these commands
from the root of a source checkout.

```bash
uv sync --locked --extra docs --extra test
uv run --no-sync python scripts/generate_manuscript_evidence.py
```

The script loads synthetic local JSON lessons, validates their course ordering,
queries an in-memory graph, and runs the same finite categorical fixture twice.
It writes the measured results to `output/data/teaching_fixture.json` and
a PNG and figure registry to `output/figures/`.

Interpret the results with the [course and graph guide](knowledge-workflow.md)
and [simulation guide](simulation-workflow.md). Use [the manuscript build
guide](manuscript-build.md) to turn the authored Markdown into PDF and HTML.

Build the documentation from the project root.

```bash
uv run --no-sync sphinx-build -W -b html docs docs/_build
```

Open `docs/_build/index.html` locally.
