import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests


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
