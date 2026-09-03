# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup -------------------------------------------------------
# Add the src directory so autodoc can find pymv
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

# -- Project information ----------------------------------------------
project = "pymv"
copyright = "2026, Abdelheq Mokhtari"
author = "Abdelheq Mokhtari"

# The full version, including alpha/beta/rc tags
import pymv

release = pymv.__version__
version = ".".join(release.split(".")[:2])  # e.g. "0.3"

# -- General configuration --------------------------------------------
extensions = [
    "sphinx.ext.autodoc",          # pull docstrings from code
    "sphinx.ext.autosummary",      # generate summary tables
    "sphinx.ext.napoleon",         # numpy/google style docstrings
    "sphinx.ext.intersphinx",      # cross-ref to numpy/sklearn docs
    "sphinx.ext.viewcode",         # [source] links
    "sphinx.ext.mathjax",          # render math
    "myst_parser",                 # .md files as sources
    "sphinx_copybutton",           # copy button on code blocks
    "sphinx_autodoc_typehints",    # pretty type hints
]

# Support both reStructuredText and Markdown
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document
master_doc = "index"

# Patterns to exclude
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Napoleon settings ------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# -- Autodoc settings -------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Avoid importing heavy deps during doc build
autodoc_mock_imports = ["tensorflow", "tf", "matplotlib", "seaborn"]

# -- Autosummary settings ---------------------------------------------
autosummary_generate = True

# -- Intersphinx mapping ----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# -- HTML output ------------------------------------------------------
html_theme = "furo"
html_title = "pymv documentation"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#2962ff",
        "color-brand-content": "#2962ff",
    },
    "dark_css_variables": {
        "color-brand-primary": "#82b1ff",
        "color-brand-content": "#82b1ff",
    },
    "source_repository": "https://github.com/AbdelheqMokhtari/pymv",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- MyST-parser settings ---------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3
