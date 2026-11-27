# Render the manuscript

The [canonical source](https://github.com/docxology/active_inference/tree/main/docs/manuscript)
uses numbered Markdown sections, paper metadata, a LaTeX preamble and BibTeX.
The teaching results and figure come from the actual local fixture.

Prepare the native artifacts from an isolated checkout whose directory is named
`active_inference` on disk so Stage 04 can find `active_inference_combined.pdf`.

```bash
uv sync --locked
uv run --no-sync python scripts/generate_manuscript_evidence.py
export PROJECT_ROOT="$(pwd -P)"
```

Prepare [docxology/template at revision
87704feae0a4490ef7dad9c931d268e8c30273c2](https://github.com/docxology/template/tree/87704feae0a4490ef7dad9c931d268e8c30273c2)
with its locked environment, Pandoc, matching pandoc-crossref, XeLaTeX and
BibTeX. Register the isolated project with the Template checkout.

```bash
export TEMPLATE_ROOT=/path/to/template
git -C "$TEMPLATE_ROOT" rev-parse HEAD
mkdir -p "$TEMPLATE_ROOT/projects/working"
ln -s "$PROJECT_ROOT" "$TEMPLATE_ROOT/projects/working/active_inference"
cd "$TEMPLATE_ROOT"
uv run --no-sync python scripts/pipeline/stage_03_render.py --project working/active_inference
uv run --no-sync python scripts/pipeline/stage_04_validate.py --project working/active_inference
```

Check that the printed Template revision matches the pin. The project
checkout receives `output/pdf/active_inference_combined.pdf`,
`output/web/index.html`, and the reports under `output/reports/`.
Review the teaching figure and citations in both rendered editions.
`rendered_provenance.json` binds the validated source, configuration, and
output artifacts.
