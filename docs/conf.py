"""Sphinx configuration shared by local builds, CI and Read the Docs."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "docs" / "_ext"))

project = "pymv"
author = "Abdelheq Mokhtari"
copyright = "2026, Abdelheq Mokhtari"
# Read the version without importing TensorFlow at configuration time.
import re
release = re.search(r'__version__ = "([^"]+)"', (ROOT / "src/pymv/__init__.py").read_text()).group(1)
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.napoleon",
    "sphinx.ext.viewcode", "sphinx.ext.mathjax", "myst_parser", "sphinx_copybutton",
    "pymv_catalog",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autodoc_mock_imports = ["tensorflow", "streamlit", "plotly"]
autodoc_default_options = {"members": True, "member-order": "bysource", "show-inheritance": True}
autodoc_typehints = "none"
autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = True
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "fieldlist"]
myst_heading_anchors = 3

html_theme = "sphinx_rtd_theme"
html_title = "pymv — multi-view clustering"
html_logo = str(ROOT / "src/pymv/assets/logo.png")
html_favicon = html_logo
html_theme_options = {
    "logo_only": False, "collapse_navigation": False, "navigation_depth": 3,
    "style_nav_header_background": "#FFFFFF", "prev_next_buttons_location": "bottom",
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
html_show_sourcelink = True
