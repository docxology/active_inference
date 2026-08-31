#!/usr/bin/env python3
"""Executable repo status — replaces prose status tables that rot.

Prints fast, verifiable facts about this repository. No network access,
no heavy imports (stdlib only, so it stays fast on external-drive checkouts).

Usage:
    python3 tools/repo_status.py            # fast facts
    python3 tools/repo_status.py --tests    # also count tests (slow on
                                            # external drives: pytest
                                            # collection can exceed 5 min)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

COMPONENT_DIRS = [
    "knowledge", "research", "applications", "platform",
    "src/active_inference", "tools", "tests", "docs",
]
STATUS_FILES = ["TODO.md", "REVIEW_LOG_2026-08-31.md", "CHANGELOG.md"]


def git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True
    )
    return out.stdout.strip() if out.returncode == 0 else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tests", action="store_true",
                    help="count collected tests via pytest (slow on external drives)")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    dirty = [l for l in git("status", "--porcelain").splitlines()
             if "__pycache__" not in l]
    info = {
        "repo": REPO.name,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "last_commit": git("log", "-1", "--format=%h %s (%cr)"),
        "sync_with_origin": git("status", "-sb").splitlines()[0] if git("status", "-sb") else "?",
        "dirty_files_excluding_pycache": len(dirty),
        "tracked_files": len(git("ls-files").splitlines()),
        "components_present": {d: (REPO / d).is_dir() for d in COMPONENT_DIRS},
        "status_files": {f: (REPO / f).exists() for f in STATUS_FILES},
        "venv_is_dangling_symlink": (
            (REPO / "venv/bin/python").is_symlink()
            and not (REPO / "venv/bin/python").exists()
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if args.tests:
        venv_py = REPO / ".venv/bin/python"
        if venv_py.exists():
            r = subprocess.run(
                [str(venv_py), "-m", "pytest", "tests", "--collect-only", "-q"],
                cwd=REPO, capture_output=True, text=True,
            )
            last = [l for l in r.stdout.splitlines() if l.strip()]
            info["pytest_collection"] = last[-1] if last else f"exit {r.returncode}, no output"
        else:
            info["pytest_collection"] = ".venv/bin/python not found"

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"repo: {info['repo']}")
        print(f"branch/sync: {info['sync_with_origin']}")
        print(f"last commit: {info['last_commit']}")
        print(f"dirty (excl. __pycache__): {info['dirty_files_excluding_pycache']}")
        print(f"tracked files: {info['tracked_files']}")
        missing = [d for d, ok in info["components_present"].items() if not ok]
        print("components:", "all present" if not missing else f"MISSING: {missing}")
        print("status files:", ", ".join(f"{f}:{'yes' if ok else 'NO'}" for f, ok in info["status_files"].items()))
        if info["venv_is_dangling_symlink"]:
            print("WARNING: venv/ is a dangling symlink — use .venv/ (see README 'Current Status')")
        if args.tests:
            print(f"pytest collection: {info['pytest_collection']}")
        print(f"(generated {info['generated_at']}; canonical backlog: TODO.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
