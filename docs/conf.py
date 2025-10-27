# Configuration file for the Sphinx documentation builder.

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath('../src'))

# Project information
project = 'Active Inference Knowledge Environment'
copyright = '2024, Active Inference Community'
author = 'Active Inference Community'
release = '0.1.0'
version = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.graphviz',
    'myst_nb',  # For markdown and Jupyter notebooks
    'nbsphinx',  # For Jupyter notebooks
    'sphinx_rtd_theme',  # Read the Docs theme
]

# MyST-NB configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# nbsphinx configuration
nbsphinx_execute = 'never'  # Don't execute notebooks during build

# Autodoc configuration
autodoc_default_flags = ['members', 'undoc-members', 'show-inheritance']
autodoc_member_order = 'bysource'

# Napoleon configuration (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'pytorch': ('https://pytorch.org/docs/stable/', None),
}

# HTML output configuration
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

# Theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# LaTeX output configuration
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

# Math configuration
math_number_all = True
math_eqref_format = 'Eq.{number}'

# Source configuration
source_suffix = {
    '.rst': None,
    '.md': None,
    '.ipynb': None,
}

# Master document
master_doc = 'index'

# Exclude patterns
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
]

# Auto summary configuration
autosummary_generate = True

# Graphviz configuration
graphviz_output_format = 'png'

# Custom configuration
def setup(app):
    """Custom setup function"""
    app.add_css_file('custom.css')

    # Add custom roles
    app.add_role('math', math_role)

def math_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Custom math role for inline math"""
    from docutils import nodes
    node = nodes.math(text=text, **options)
    return [node], []
