# Contributing

Keep operational instructions under [docs](README.md) and the paper sources
under `docs/manuscript/`. Retain conceptual lessons and machine-readable
examples in their source trees. Component README files can point readers
to the appropriate guide.

Run the checks relevant to a documentation or teaching example change from
the repository root.

```bash
uv sync --locked --extra test --extra docs
uv run --no-sync python -m pytest tests/unit/test_knowledge_repository.py tests/unit/test_platform_knowledge_graph.py tests/unit/test_cov_research_simulations.py
uv run --no-sync python scripts/generate_manuscript_evidence.py
uv run --no-sync python scripts/check_ratchets.py sphinx
```

Use the repository's existing Python test and style gates for changes to the
runtime. Include the actual check results in your pull request. For manuscript
edits, follow the [pinned rendering workflow](manuscript-build.md).

The [Code of Conduct](https://github.com/docxology/active_inference/blob/main/docs/code_of_conduct.rst)
applies to contributions.
