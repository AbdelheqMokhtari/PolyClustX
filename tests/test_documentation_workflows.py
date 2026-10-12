"""Keep the published examples runnable and optional screenshots unobtrusive."""

import io
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_practical_workflows_create_successful_reports_and_figures(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    workflows = runpy.run_path(str(ROOT / "examples/practical_workflows.py"))
    workflows["main"](tmp_path)
    reports = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert len(reports) == 5  # comparison, unlabeled K study, and three searches
    for report in reports:
        assert report["experiments"]
        for experiment in report["experiments"]:
            for run in experiment["runs"]:
                assert run["status"] == "complete", run.get("error")
                assert all(value is not None and np.isfinite(value) for value in run["metrics"].values())
    for filename in ("comparison.zip", "clusters.png", "losses.svg"):
        assert (tmp_path / filename).stat().st_size > 100


def test_sidebar_and_optional_screenshot_in_sphinx(tmp_path):
    sphinx = pytest.importorskip("sphinx.application")
    pytest.importorskip("sphinx_rtd_theme")
    source = tmp_path / "source"
    source.mkdir()
    (source / "conf.py").write_text(
        f"import sys\nsys.path.insert(0, {str(ROOT / 'docs/_ext')!r})\n"
        "extensions = ['polyclustx_catalog']\n"
        "html_theme = 'sphinx_rtd_theme'\n"
        f"templates_path = [{str(ROOT / 'docs/_templates')!r}]\n"
        "html_static_path = ['_static']\n"
        "html_context = {'polyclustx_css': ''}\n",
        encoding="utf-8",
    )
    screenshots = source / "_static/screenshots"
    screenshots.mkdir(parents=True)
    (source / "index.rst").write_text(
        "Interface\n=========\n\n.. polyclustx-screenshot:: interface-overview.png\n\n"
        "   The dataset page.\n", encoding="utf-8",
    )
    warnings = io.StringIO()
    app = sphinx.Sphinx(str(source), str(source), str(tmp_path / "html"),
                        str(tmp_path / "doctrees"), "html", status=io.StringIO(), warning=warnings)
    app.build()
    assert not warnings.getvalue()
    page = (tmp_path / "html/index.html").read_text()
    assert page.count('class="pcx-affiliations"') == 1
    assert page.index('class="pcx-affiliations"') < page.index('class="wy-nav-content-wrap"')
    assert "Screenshot guide" not in page and "Image reference:" not in page
    assert "interface-overview.png" not in page

    # A supplied capture appears on an incremental rebuild without editing the guide.
    shutil.copyfile(ROOT / "src/polyclustx/display/assets/logo.png", screenshots / "interface-overview.png")
    app.build()
    assert not warnings.getvalue()
    page = (tmp_path / "html/index.html").read_text()
    assert "interface-overview.png" in page and "The dataset page." in page
    assert (tmp_path / "html/_images/interface-overview.png").is_file()
