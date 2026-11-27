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

## Workflowz review — 2026-11-10

### Major

- [ ] **WF-20261110-01 (P0): Ship every package submodule in built distributions.** Replace the root-only `packages = ["active_inference"]` declaration with setuptools package discovery so wheels contain `knowledge`, `research`, `llm`, `platform`, and the other public subpackages, then add a wheel-content smoke test. Evidence: `pyproject.toml:69-71`; public submodule exports at `src/active_inference/knowledge/__init__.py:1-34` and `src/active_inference/research/__init__.py:1-35`. Acceptance (repo root): `uv build --wheel && python3 -c "import glob,zipfile; p=glob.glob('dist/*.whl')[-1]; n=zipfile.ZipFile(p).namelist(); assert 'active_inference/knowledge/__init__.py' in n and 'active_inference/research/__init__.py' in n"` exits 0. Dependencies: none; retain the `src` layout.
- [ ] **WF-20261110-02 (P0): Make every installed console script resolve to a real callable.** `ai-knowledge`, `ai-research`, and `ai-visualize` currently target package `__init__.py` modules that define no `main`; either route the aliases through working command implementations or remove them rather than shipping commands that fail on startup. Evidence: `pyproject.toml:51-54`; the only production `main` definitions are `src/active_inference/cli.py:869-873` and `src/active_inference/tools/documentation/cli.py:444`. Acceptance (repo root): `uv sync --extra test && uv run python -c "import importlib.metadata as m; eps=[e for e in m.entry_points(group='console_scripts') if e.name in {'active-inference','ai-knowledge','ai-research','ai-visualize'}]; [e.load() for e in eps]"` succeeds, and each retained script's `--help` exits 0. Dependencies: WF-20261110-01 for meaningful non-editable wheel checks; CLI routing must remain in one implementation.
- [ ] **WF-20261110-03 (P0): Stop reporting synthetic experiment trajectories as completed research.** `ExperimentManager.run_experiment` defaults to `_generate_mock_results`, and `ResearchFramework.run_study` uses that default, so normal orchestration writes fabricated data to `results.json`; make real execution the production path, isolate any demonstration generator under an explicit demo/test API, and honor `ExperimentConfig.random_seed` with a local NumPy generator. Evidence: `src/active_inference/research/experiments.py:76-103`, `src/active_inference/research/experiments.py:200-233`, and `src/active_inference/research/experiments.py:296-306`. Acceptance (repo root): add `tests/unit/test_research_experiments.py` covering real `run_study` output, explicit demo labeling, and repeatable same-seed/different-seed behavior; `uv run python -m pytest tests/unit/test_research_experiments.py` passes and no default production call reaches `_generate_mock_results`. Dependencies: none; do not silently relabel existing synthetic output as measured results.
- [ ] **WF-20261110-04 (P0): Replace `DataManager` placeholder success with real collection, validation, and integrity behavior.** Source collectors return literal placeholder dictionaries, preprocessing and quality checks are no-ops, format validation always succeeds, HDF5/custom storage silently writes nothing, and size is always 1024; implement supported paths and raise explicit errors for unsupported paths instead of returning successful metadata. Update tests that currently encode no-op behavior and fail to assert missing-value/outlier results. Evidence: `src/active_inference/research/data_management.py:137-161`, `:163-245`, `:248-280`, `:282-311`; `tests/unit/test_research_data_management.py:104-128`, `:147-170`. Acceptance (repo root): `uv run python -m pytest tests/unit/test_research_data_management.py` passes new assertions for real normalization/filtering, invalid formats, missing values, outliers, actual byte size/checksum, persisted retrieval, and explicit unsupported-source/storage failures. Dependencies: none; only advertise formats and sources with actual implementations.
- [ ] **WF-20261110-05 (P0): Return deterministic, mathematically defined expected free energy values.** `Mathematics.expected_free_energy` ignores both policy and observation contents and returns random numbers in both NumPy and pure-Python branches; implement the documented risk/ambiguity/value calculation (or narrow the API to inputs it genuinely uses) and test known numerical cases plus input validation. Evidence: `src/active_inference/knowledge/mathematics.py:313-343`; related component arithmetic already exists and is tested at `tests/unit/test_knowledge_mathematics.py:98-115`. Acceptance (repo root): `uv run python -m pytest tests/unit/test_knowledge_mathematics.py` passes new exact-value tests, repeated calls with identical inputs are equal, and changing a policy or observation changes the result when mathematically expected. Dependencies: none; preserve the NumPy-optional branch or remove it consistently across the module.
- [ ] **WF-20261110-06 (P0): Correct the pure-Python KL divergence formula.** The no-NumPy branch computes `sum(p * (p/q))` rather than `sum(p * log(p/q))`, disagreeing with the NumPy branch; use the logarithmic expression and explicitly handle zero probabilities and mismatched/invalid distributions. Evidence: `src/active_inference/knowledge/mathematics.py:284-311`. Acceptance (repo root): add parity tests for both branches against a hand-computed distribution and edge cases; `uv run python -m pytest tests/unit/test_knowledge_mathematics.py` passes. Dependencies: none; do not hide numerical errors by adding a tolerance large enough to accept the incorrect formula.

### Medium

- [ ] **WF-20261110-07 (P1): Persist mathematics nodes instead of discarding them during initialization.** `Mathematics.__init__` builds all four node groups, but `_save_nodes_to_repository` is `pass`, unlike the working `Foundations` implementation; reuse the repository's public insertion path (or the established `KnowledgeNode` conversion pattern) and add a regression test that initialized concepts are retrievable. Evidence: `src/active_inference/knowledge/mathematics.py:35-43`, `:248-252`; working comparison at `src/active_inference/knowledge/foundations.py:240-258`. Acceptance (repo root): `uv run python -m pytest tests/unit/test_knowledge_mathematics.py` passes assertions that `probability_basics` and the other initialized IDs exist and contain valid `KnowledgeNode` content. Dependencies: none; avoid broad exception suppression when malformed nodes fail validation.
- [ ] **WF-20261110-08 (P1): Make API integrations perform I/O or fail explicitly.** `IntegrationManager.connect_to_api` marks any URL connected without a request and hard-codes a timestamp, while `query_api` returns `mock_data` with success; implement HTTP connection/query behavior using the declared `httpx` dependency and connector timeout/auth configuration, or reject unsupported operations without manufacturing success. Evidence: `src/active_inference/applications/integrations.py:62-102`; dependency declaration at `pyproject.toml:29-35`. Acceptance (repo root): add `tests/unit/test_applications_integrations.py` with mocked transport asserting real URL/parameters/headers/timeout and non-success behavior; `uv run python -m pytest tests/unit/test_applications_integrations.py` passes and the production literal `"mock_data"` is gone. Dependencies: none; transport test doubles belong in tests, not the return path.
- [ ] **WF-20261110-09 (P1): Do not exit successfully for unimplemented CLI operations.** Research, visualization, applications, platform, and data handlers print “coming soon” or claim validation/backup actions while returning 0; connect them to real implementations or return a documented nonzero unavailable-command status and remove claims that an action occurred. Evidence: `src/active_inference/cli.py:464-496` and `:852-866`. Acceptance (repo root): add CLI subprocess tests for every advertised command; `uv run python -m pytest tests/unit/test_cli.py` passes, implemented commands exit 0, and unavailable commands exit nonzero without success messages. Dependencies: WF-20261110-02 for installed-script coverage; no network service should be required for unavailable-command tests.
- [ ] **WF-20261110-10 (P1): Repair or remove dead Makefile entry points.** Numerous targets activate the retired `venv/` environment and invoke absent scripts (`serve-knowledge`, learning tracks, research runners, visualization, and application generators), despite setup using `uv`; use `uv run` and existing entry points, and delete targets whose implementations do not exist. Evidence: `Makefile:84-175`; environment convention at `Makefile:29-45` and `TODO.md` Completed Round 2; available platform file inventory includes `platform/serve.py` but not the additional server scripts. Acceptance (repo root): `make lint && make docs` completes using uv-managed tools, and `python3 -c "from pathlib import Path; assert not Path('platform/knowledge_server.py').exists()"` documents why removed aliases must not remain; every retained Python-target path must exist. Dependencies: documentation commands need their Sphinx dependencies declared/reused consistently; do not recreate placeholder scripts merely to satisfy Make.
- [ ] **WF-20261110-11 (P1): Make security failures fail CI.** Both repository security tests and the dependency vulnerability scan use `continue-on-error: true`, so the security job can pass after either check fails; remove those tolerances and declare the scanner in development dependencies instead of installing an unpinned tool during the job. Evidence: `.github/workflows/ci.yml:84-109`. Acceptance (repo root): `uv run python -m pytest tests/security/ -v --tb=short` and `uv run safety check` are the exact non-tolerated CI commands, and a deliberately failing local test returns nonzero without the workflow masking it. Dependencies: triage any currently detected vulnerability explicitly; never resolve this by suppressing the finding.
- [ ] **WF-20261110-12 (P1): Stop converting package import defects into `None` public APIs.** The root package wraps whole submodule import lists in broad `except ImportError`, so a missing transitive dependency or an import-time defect silently replaces classes with `None`; use direct imports for required package dependencies or narrowly scoped optional imports with an explicit availability error at use. Evidence: `src/active_inference/__init__.py:21-41`, `:44-70`, and the repeated blocks through `:134`. Acceptance (repo root): add an import test proving public classes are not `None` and that an induced submodule `ImportError` retains its module/error context; `uv run python -m pytest tests/unit/test_package_imports.py` passes. Dependencies: WF-20261110-01 ensures the same behavior is testable from a wheel; retain lazy imports only if they measurably address the existing collection-speed item.

### Minor

- [ ] **WF-20261110-13 (P2): Align package metadata and tool interpreter targets with the supported runtime.** Distribution metadata says 0.3.0 while `active_inference.__version__` says 0.1.0; Black and mypy still target Python 3.9 although project metadata/lock support `>=3.11,<3.13` and CI tests 3.11/3.12. Use one version source and set static-tool targets to the actual floor, without blindly expanding the runtime range. Evidence: `pyproject.toml:6-7`, `:14-26`, `:75-76`, `:144-146`; `src/active_inference/__init__.py:13-15`; `uv.lock:2-6`; `.github/workflows/ci.yml:17-26`. Acceptance (repo root): `uv run python -c "import active_inference,importlib.metadata as m; assert active_inference.__version__ == m.version('active-inference-knowledge')"` exits 0, and `uv run black --check src/ tests/ && uv run mypy src/` runs under the declared 3.11 target. Dependencies: coordinate with WF-20261110-02 so versioned console-script metadata is regenerated; no Python 3.15.2 migration is proposed without resolving the existing locked scientific-dependency range.
