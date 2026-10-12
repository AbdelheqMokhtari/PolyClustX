"""Helpers shared by the dataset tests."""

from __future__ import annotations

import pytest
import requests

from polyclustx.data import load_dataset


def load_or_skip(name: str):
    """Load a dataset by name, skipping the test if Zenodo is unreachable.

    Read timeouts and dropped connections say nothing about the code under
    test, so they skip rather than fail.  A dataset that downloads but does
    not parse still raises, which is the failure worth seeing.
    """
    try:
        return load_dataset(name)
    except requests.RequestException as exc:
        pytest.skip(f"could not download '{name}': {type(exc).__name__}: {exc}")
