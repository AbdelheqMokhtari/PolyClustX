"""Sphinx configuration shared by local builds, CI and Read the Docs."""
import hashlib
import os
import re
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "docs" / "_ext"))

identity = runpy.run_path(str(ROOT / "src/polyclustx/brand.py"))
project = identity["NAME"]
author = "Abdelheq Mokhtari"
copyright = "2026, Abdelheq Mokhtari"
# Read the version without importing the numerical backend at configuration time.
release = re.search(r'__version__ = "([^"]+)"', (ROOT / "src/polyclustx/__init__.py").read_text()).group(1)
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.napoleon",
    "sphinx.ext.viewcode", "sphinx.ext.mathjax", "myst_parser", "sphinx_copybutton",
    "polyclustx_catalog",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autodoc_mock_imports = ["torch", "streamlit", "plotly"]
autodoc_default_options = {"members": True, "member-order": "bysource", "show-inheritance": True}
autodoc_typehints = "none"
autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = True
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "fieldlist"]
myst_heading_anchors = 3

html_theme = "sphinx_rtd_theme"
html_title = f"{project} — From Feature Views to Attributed Graphs"
LOGO_PATH = ROOT / "src/polyclustx/display/assets/logo.png"
html_logo = str(LOGO_PATH)
html_favicon = html_logo
html_theme_options = {
    "logo_only": False, "collapse_navigation": False, "navigation_depth": 3,
    "style_nav_header_background": "#FFFFFF", "prev_next_buttons_location": "bottom",
}
html_static_path = ["_static", str(ROOT / "src/polyclustx/display/assets")]
html_css_files = ["custom.css"]
templates_path = ["_templates"]
html_context = {
    "polyclustx_definition": identity["DEFINITION"],
    "polyclustx_css": identity["css_variables"](),
    # Also invalidate incremental HTML builds when the shared logo changes.
    "polyclustx_logo_version": hashlib.sha256(LOGO_PATH.read_bytes()).hexdigest()[:12],
}
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
html_show_sourcelink = True


def _version_brand_images(app, pagename, templatename, context, doctree):
    """Refresh cached brand images without duplicating the package assets."""
    if app.builder.name not in {"html", "dirhtml", "singlehtml"}:
        return
    for key in ("logo_url", "favicon_url"):
        url = context.get(key)
        if url:
            separator = "&" if "?" in url else "?"
            context[key] = f"{url}{separator}v={context['polyclustx_logo_version']}"


def setup(app):
    # Run after Sphinx has resolved resource paths for the current page.
    app.connect("html-page-context", _version_brand_images, priority=600)
