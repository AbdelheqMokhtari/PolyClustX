"""Every registered algorithm follows the same package layout."""

import importlib
from pathlib import Path

from polyclustx.models import list_models_info


def test_registered_estimators_live_in_uniform_packages():
    for name, info in list_models_info().items():
        package = importlib.import_module(f"polyclustx.models.{name}")
        implementation = importlib.import_module(f"polyclustx.models.{name}.{name}")
        assert Path(package.__file__).name == "__init__.py"
        assert info["class"].__module__ == implementation.__name__
        assert getattr(package, info["class"].__name__) is info["class"]
