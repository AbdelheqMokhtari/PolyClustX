"""Protect the public rename, shared identity and unexecuted tutorials."""

import ast
import json
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import plotly.graph_objects as go
from PIL import Image

import polyclustx
from polyclustx.brand import BRAND, DEFINITION, NAME, PLOT_COLORS, css_variables
from polyclustx.data.base import get_data_home
from polyclustx.display.theme import CSS, style_figure
from polyclustx.experiment.exports import new_report

ROOT = Path(__file__).resolve().parents[1]


def test_distribution_and_public_namespace():
    package = distribution("polyclustx")
    assert package.metadata["Name"] == "polyclustx"
    scripts = {entry.name: entry.value for entry in package.entry_points}
    assert scripts["polyclustx"] == "polyclustx.command:main"
    assert scripts["polyclustx-display"] == "polyclustx.display:launch"
    assert "polyclustx-cli" not in scripts
    assert not any(name.startswith("pymv") for name in scripts)
    assert len(polyclustx.list_models()) == 13
    assert polyclustx.OptunaSearchExperiment.__module__.startswith("polyclustx.")
    assert not (ROOT / "src/pymv").exists()
    for source in (ROOT / "src/polyclustx").rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                assert all(not name.name.startswith("pymv") for name in node.names), source
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("pymv"), source


def test_module_cli_uses_new_name():
    result = subprocess.run([sys.executable, "-m", "polyclustx", "--help"],
                            text=True, capture_output=True, check=True, timeout=60)
    assert "polyclustx" in result.stdout
    assert "display" in result.stdout
    assert "pymv" not in result.stdout


def test_cache_and_report_names(monkeypatch, tmp_path):
    monkeypatch.setenv("POLYCLUSTX_DATA_DIR", str(tmp_path / "cache"))
    assert get_data_home() == tmp_path / "cache"
    report = new_report()
    assert report["polyclustx_version"] == polyclustx.__version__
    assert "pymv_version" not in report


def test_shared_palette_and_project_definition():
    assert NAME == "PolyClustX"
    assert DEFINITION == (
        "A Unified Framework for Multi-View and Multimodal Representation Learning "
        "and Clustering from Feature Views to Attributed Graphs"
    )
    assert BRAND == {
        "primary": "#4E4E92", "primary_dark": "#3B3D73", "primary_light": "#7478AE",
        "secondary": "#634E87", "accent": "#557AA3", "background": "#F7F8FC",
        "surface": "#FFFFFF", "surface_alt": "#EEF0F7", "border": "#D7DAE8",
        "text": "#292B3A", "text_muted": "#6B6E7D", "brand_tint": "#E8E9F3",
    }
    assert CSS.startswith(css_variables())
    assert "#8A1538" not in CSS
    figure = go.Figure(go.Scatter(x=[1, 2], y=[2, 1]))
    assert style_figure(figure) is figure
    assert figure.layout.paper_bgcolor == BRAND["surface"]
    assert figure.layout.font.color == BRAND["text"]
    assert list(figure.layout.colorway) == PLOT_COLORS
    assert figure.layout.xaxis.gridcolor == BRAND["border"]


def test_transparent_logo_and_institutional_assets():
    assets = ROOT / "src/polyclustx/display/assets"
    with Image.open(assets / "logo.png") as logo:
        assert logo.mode == "RGBA"
        assert logo.getchannel("A").getextrema() == (0, 255)
        assert logo.width == logo.height
    for name in ("borelli.png", "paris_cite.png"):
        with Image.open(assets / name) as logo:
            logo.verify()
    conf = (ROOT / "docs/conf.py").read_text()
    assert 'src/polyclustx/brand.py' in conf
    assert 'src/polyclustx/display/assets' in conf
    sidebar = (ROOT / "docs/_templates/affiliations.html").read_text()
    assert "borelli.png" in sidebar and "paris_cite.png" in sidebar
    layout = (ROOT / "docs/_templates/layout.html").read_text()
    assert '{% block navigation %}\n{% include "affiliations.html" %}' in layout
    assert not (ROOT / "docs/_templates/footer.html").exists()


def test_notebooks_are_valid_unexecuted_new_api_tutorials():
    """Inspect source and structure only: never execute a notebook cell."""
    notebooks = list((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) >= 7
    for path in notebooks:
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            assert cell["execution_count"] is None, path
            assert cell["outputs"] == [], path
            source = "".join(cell["source"])
            assert "from pymv" not in source and "import pymv" not in source, path
            # Older tutorials may contain IPython line magics; syntax-check only Python.
            python = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
            ast.parse(python, filename=str(path))
    study = (ROOT / "notebooks/06_polyclustx_feature_to_graph_study.ipynb").read_text()
    for symbol in ("load_iris", "load_wine", "Experiment", "plot_clusters", "plot_loss_comparison",
                   "GridSearchExperiment", "RandomSearchExperiment", "OptunaSearchExperiment"):
        assert symbol in study
