# TODO.md — Backlog (canonical)

> **Single source of truth for open work in this repo.** README.md "Current
> Status" links here and must never copy this roster. Append-only: move
> finished items to Completed with a date; never delete.

*Created 2026-08-31 by the agent-ergonomics deep pass; Round 2 executed the
same day. Every entry records file path(s) and, where relevant, the command
that verifies the claim.*

## Open

### Minor

- [ ] Review delinked references: 501 dead link targets were converted to
  inline code spans (path text preserved) across the repo. Owners should
  decide, tree by tree, whether each referenced artifact should be created or
  the reference dropped. Search: `rg '\`[a-z_./]+\.md\`' --type md`. (repo-wide)
- [ ] tests/README.md and per-suite AGENTS/README files still describe
  expectations (>95% coverage etc.) that are not enforced by any gate in this
  repo — either add a gate or soften the claims. (tests/README.md, AGENTS.md)
- [ ] Root README "Quality Gates" section claims ">95% core / >80% overall"
  coverage; no coverage gate is wired in this repo (pytest.ini has no
  --cov-fail-under; Makefile test-all does, but it is not the default and
  could not be run on the external drive). Verify and align or implement.
  (README.md, Makefile)

### Medium

- [ ] **Test gate speed.** pytest collection on external-drive checkouts
  exceeds 5 minutes (measured 2026-08-31, see REVIEW_LOG_2026-08-31.md);
  `import pytest` alone ≈47s. Options: lazy imports in
  `src/active_inference/__init__.py` (it eagerly imports all submodules),
  running the venv from internal disk, or accepting and documenting the cost
  (now documented in README/AGENTS/QUICK_START). Verify after fixing with:
  `time .venv/bin/python -m pytest tests/unit --collect-only -q`.
- [ ] **README/AGENTS dedup.** AGENTS.md still repeats the README component
  catalog (~600 lines). Keep one canonical description (README), reduce
  AGENTS.md to agent conventions + links. Owner call on how much narrative to
  keep. (AGENTS.md)
- [ ] **Aspirational quality claims.** "99.9% uptime", "sub-second response
  times" and similar platform promises in README/AGENTS describe a target, not
  a measured state. Rewrite as goals or delete. (README.md, AGENTS.md)

### Major

- [ ] **README is still an aspirational catalog after the Round-1/2 fixes.**
  The structure tree and top-of-file status are now real, but the component
  sections promise capabilities that are unverified (LLM integration,
  dashboards, 3D exploration). A full rewrite needs the owner to say what is
  actually supported. Acceptance: every capability claim in README carries a
  verification path (command or demo) or is marked as planned.
  (README.md, TODO.md)

## Completed — Round 1 (2026-08-31)

- [x] Create TODO.md as canonical backlog. (this file)
- [x] README "Current Status" section with measured evidence and verification
  paths. (README.md)
- [x] Archive 9 transient summary reports + 5 audit JSONs under `_archive/`
  (+`_archive/audits/`) with explanatory README; no live doc links to them.
- [x] REVIEW_LOG_2026-08-31.md dated log of the pass and measurements.
- [x] venv situation (dangling `venv/` symlink; use `.venv/`) recorded in
  status surfaces.

## Completed — Round 2 (2026-08-31)

- [x] **91→0 broken relative links.** Existence-checked mechanical pass over
  all 332 tracked .md files: 299 links re-pointed (one-level-depth fix or
  repo-root-relative), 501 dead targets converted to inline code spans
  preserving the path. Verified by re-running the checker: 0 broken remain.
  Commit 0488041. (157 files)
- [x] Root `LICENSE` file added (MIT — matches pyproject `license = {text =
  "MIT"}`); README/AGENTS license sentence re-linked. (LICENSE, README.md,
  AGENTS.md)
- [x] Code of Conduct reference re-pointed at the existing
  `docs/code_of_conduct.rst`. (CONTRIBUTING.md)
- [x] AGENTS.md Branch Strategy corrected: trunk-based `main`; no `develop`
  branch exists (remote verified). (AGENTS.md)
- [x] README structure tree replaced with the actual on-disk tree (root
  `visualization/` and phantom `tools/` subdirs removed; note added pointing
  at `src/active_inference/visualization/` and `research/visualization/`).
  (README.md)
- [x] All `visualization/README.md`-style refs in README/AGENTS re-pointed at
  existing `research/visualization/` files. (README.md, AGENTS.md)
- [x] docs/DOCUMENTATION_INDEX.md `../../` links rewritten to `../` (16 links;
  0 broken remain in the file). (docs/DOCUMENTATION_INDEX.md)
- [x] venv guidance corrected to uv-managed `.venv/` in README, AGENTS.md, and
  docs/QUICK_START.md; external-drive collection latency documented at each
  `make test` mention. (README.md, AGENTS.md, docs/QUICK_START.md)
- [x] `tools/repo_status.py` — executable status surface (stdlib-only, fast;
  `--tests` flag for pytest count; `--json` for machines). Documented in
  tools/README.md and README Current Status. (tools/repo_status.py,
  tools/README.md, README.md)
- [x] Root generated-audit JSONs gitignored (archived copies retained under
  `_archive/audits/`). (.gitignore)

## Re-scoped / deferred (owner intent required)

- README full rewrite (Major, above) — needs owner's statement of what is
  actually supported vs planned.
- README/AGENTS dedup depth — owner call on narrative retention.
- Test-gate speed root fix (lazy imports) — code change beyond doc scope;
  acceptance line recorded above.
