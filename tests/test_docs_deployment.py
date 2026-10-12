import hashlib
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests


@pytest.mark.parametrize("prefix", ["_static/", "../_static/", "../../../_static/"])
def test_docs_logo_and_favicon_use_shared_content_version(prefix):
    root = Path(__file__).resolve().parents[1]
    config = runpy.run_path(str(root / "docs/conf.py"))
    logo = root / "src/polyclustx/display/assets/logo.png"
    version = hashlib.sha256(logo.read_bytes()).hexdigest()[:12]
    assert config["html_logo"] == str(logo)
    assert config["html_context"]["polyclustx_logo_version"] == version
    context = {
        **config["html_context"],
        "logo_url": f"{prefix}logo.png",
        "favicon_url": f"{prefix}logo.png",
    }
    config["_version_brand_images"](
        SimpleNamespace(builder=SimpleNamespace(name="html")),
        "index", "page.html", context, None,
    )
    assert context["logo_url"] == f"{prefix}logo.png?v={version}"
    assert context["favicon_url"] == context["logo_url"]


def test_deploy_requires_credentials_and_valid_slugs(monkeypatch):
    deploy = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/deploy_docs.py"))["deploy"]
    with pytest.raises(ValueError, match="READTHEDOCS_TOKEN"):
        deploy("polyclustx", "latest", None)
    with pytest.raises(ValueError, match="slugs"):
        deploy("../invalid", "latest", "test-token")
    calls = []
    monkeypatch.setattr(requests, "post", lambda url, **kwargs:
                        (calls.append((url, kwargs)) or SimpleNamespace(status_code=202)))
    assert deploy("polyclustx", "latest", "test-token") == "https://app.readthedocs.org/projects/polyclustx/builds/"
    assert calls[0][0].endswith("/versions/latest/builds/")
    assert calls[0][1]["headers"]["Authorization"] == "Token test-token"
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: SimpleNamespace(status_code=403))
    with pytest.raises(ValueError, match="HTTP 403"):
        deploy("polyclustx", "latest", "test-token")
