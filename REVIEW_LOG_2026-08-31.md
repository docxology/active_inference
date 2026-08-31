# REVIEW_LOG_2026-08-31.md — Agent-Ergonomics Deep Pass

Append-only dated log. Newest entries at the bottom.

## 2026-08-31 — Agent-ergonomics fleet pass (active-inference lane)

Branch: main. Pre-existing dirty files at dispatch: 6 (untracked .github/* and
docs/manuscript/* AGENTS/README pairs — untouched by this pass).

### Cold-start audit score (entry docs only, before changes)

- (a) Determine current project status: **FAIL** — README/AGENTS describe an
  aspirational platform ("99.9% uptime", ">95% coverage") with no status
  section, no backlog pointer, and no verification path.
- (b) Find what to do next: **FAIL** — no backlog file existed (TODO.md
  created this pass).
- (c) Find how to run primary verification: **PARTIAL FAIL** — `make test` is
  documented, but the documented default venv path (`venv/`) is a dangling
  symlink to miniforge; the working env is `.venv/` (uv-managed), and pytest
  collection there is extremely slow on this external-drive checkout.

### Verification performed this pass (all measured, not claimed)

- `git fetch origin` — repo up to date with origin/main before work.
- Link check script over README.md, AGENTS.md, CONTRIBUTING.md,
  docs/README.md, docs/DOCUMENTATION_INDEX.md, tests/README.md,
  src/README.md: **91 broken relative links** (dominant causes: root
  `visualization/` does not exist; `docs/DOCUMENTATION_INDEX.md` uses `../../`
  paths; `LICENSE` absent; tools/ subdirs documented but absent).
- `venv/bin/python` is a dangling symlink (`-> /Users/4d/miniforge3/bin/python3.12`,
  target missing). The functional environment is `.venv/` (uv-managed,
  Python 3.12.13, pytest 8.4.2).
- Import-cost timing in `.venv/bin/python`: `import pytest` ≈ 47.5 s;
  `import tests` ≈ 33.2 s; `import yaml` ≈ 5.9 s; `import pydantic` ≈ 18.0 s;
  `import active_inference` and `pytest --collect-only tests/unit` exceed
  5 minutes and did not finish (killed at timeout). One full-file pytest run
  of tests/unit/test_knowledge_foundations.py ran >15 min without completing
  and was killed. Cause is I/O latency of the external drive plus heavy
  import chains (pydantic/yaml in knowledge/repository.py; package
  __init__ eagerly imports all submodules).
- `git ls-files`: 8016 tracked files, 329 tracked .md files.

### Changes made this pass

1. Created TODO.md (canonical backlog; all findings filed as Minor/Medium/Major).
2. Archived 9 transient summary/report .md files and 5 audit JSONs into
   `_archive/` (+`_archive/audits/`) with an explanatory README; removed from
   repo root. No live doc links to them.
3. Added "Current Status" section to README.md (links TODO.md and this log;
   states verification reality including the dangling venv symlink).
4. Created REVIEW_LOG_2026-08-31.md (this file).
5. Recorded the venv situation and slow-test reality in TODO.md (Medium).

### Deferred

- Full README rewrite (Major, TODO.md): 950-line aspirational catalog with a
  structure tree listing non-existent directories. Too large to rewrite
  safely in this pass without touching claims I cannot verify; scoped in TODO.
- README/AGENTS deduplication (Medium): same reason.
- Link repair at scale (91 links, many in docs/ trees): requires per-tree
  decisions about which documented-but-absent dirs should exist vs. be
  un-documented; filed as Minor items.
- No test-gate pass is claimed: pytest collection does not complete in
  reasonable time on this drive (see timing above).
