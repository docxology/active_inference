"""Configuration for the maintained Sphinx documentation."""

from pathlib import Path
from tomllib import load

with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as stream:
    metadata = load(stream)["project"]

project = "Active Inference Knowledge Environment"
author = "Active Inference Community"
release = metadata["version"]
version = release
root_doc = "index"

extensions = ["myst_parser", "sphinx.ext.mathjax"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# The teaching site includes the current operational guides. Authored notes
# elsewhere in docs remain available in the source repository.
include_patterns = [
    "index.rst",
    "README.md",
    "QUICK_START.md",
    "knowledge-workflow.md",
    "simulation-workflow.md",
    "manuscript-build.md",
    "CONTRIBUTING.md",
]
exclude_patterns = ["_build", ".doctrees"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
