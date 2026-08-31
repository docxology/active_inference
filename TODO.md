# TODO.md — Backlog (canonical)

> **Single source of truth for open work in this repo.** README.md "Current
> Status" links here and must never copy this roster. Append-only: move
> finished items to Completed with a date; never delete.

*Created 2026-08-31 by the agent-ergonomics deep pass. Every Phase-1
cold-start finding has an entry. Update when you finish an item.*

## Minor

- [ ] README.md "Complete Project Structure" tree (~lines 357-620) inventories `visualization/`, per-domain application READMEs, and sub-READMEs that do not exist on disk — trim to actual directories. (README.md)
- [ ] Dead links from README.md/AGENTS.md: `visualization/README.md`, `visualization/AGENTS.md`, root `LICENSE` (pyproject declares MIT but no LICENSE file is tracked). (README.md, AGENTS.md)
- [ ] AGENTS.md "Branch Strategy" claims a `develop` branch + PR flow; the remote has only `main` and work is trunk-based — fix the doc or the practice. (AGENTS.md)
- [ ] Generated audit JSONs at root (documentation_audit_report.json, knowledge_audit_report.json, interactive_exercises_report.json, metadata_completion_report.json, multimedia_integration_report.json) have no gitignore or archive home — move under `_archive/audits/`. (root)

## Medium

- [ ] **Test gate is not runnable as documented.** pytest 8.4.2 is installed in `.venv`, but `import pytest` there takes ~47s on this external-drive checkout and `pytest tests/unit --collect-only` exceeded 5 minutes without finishing (measured 2026-08-31, see REVIEW_LOG_2026-08-31.md). `make test` will appear to hang. Fix imports in tests/__init__.py / src, or run the venv from internal disk, or document the expected multi-minute wait everywhere `make test` is mentioned. (Makefile, tests/__init__.py, pyproject.toml)
- [ ] README has no "Current Status" section — a cold agent cannot tell real state from aspiration in 950 lines. Add a short status block linking TODO.md and REVIEW_LOG_2026-08-31.md. (README.md)
- [ ] AGENTS.md duplicates the README component catalog instead of linking it; keep one canonical description (README) and reduce AGENTS.md to agent conventions + links. (AGENTS.md)
- [ ] Nine transient audit/summary .md reports at root (COMPLETE_MAINTENANCE_SUMMARY, CONTINUING_MAINTENANCE_SUMMARY, CROSS_REFERENCE_AUDIT_COMPLETE, CROSS_REFERENCE_FINAL_SUMMARY, DOCUMENTATION_AUDIT_REPORT, ENHANCEMENT_SUMMARY, FINAL_COMPLETION_REPORT, MAINTENANCE_COMPLETE_SUMMARY, documentation_audit_summary) read as current state but describe past passes — archive under `_archive/` and drop from live navigation. (root)
- [ ] docs/DOCUMENTATION_INDEX.md uses `../../` relative paths that are broken from its location — fix or relocate. (docs/DOCUMENTATION_INDEX.md)

## Major

- [ ] **Root README is an aspirational catalog, not documentation.** 950 lines; the structure tree lists directories that don't exist. A cold agent builds a wrong mental model in the first 2 minutes. Rewrite to lead with: what this is (2 lines) -> current status (link TODO.md) -> how to verify (`make test` + import-cost caveat) -> component map (links only, no duplicated prose). (README.md)
- [ ] **No executable status surface.** Nothing prints repo state (last commit, test counts, which advertised component dirs exist). A tiny `tools/repo_status.py` would beat prose; deferred as not trivial to make both fast and useful on this drive. (tools/)

## Completed

- (none yet)
