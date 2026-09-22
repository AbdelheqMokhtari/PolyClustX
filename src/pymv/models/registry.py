"""
Model registry for pymv.

Provides a lightweight decorator-based system for registering
multi-view clustering models.  Adding a new model only requires:

1. Creating the class (subclass of ``BaseModel``).
2. Decorating it with ``@register_model("name")``.

No modification to core framework code is needed.

Example
-------
>>> from pymv.models.registry import register_model, get_model, list_models
>>>
>>> @register_model("my_model")
... class MyModel(BaseModel):
...     def fit(self, X): ...
>>>
>>> list_models()
['lmgec', 'my_model']
>>>
>>> cls = get_model("my_model")
>>> model = cls(n_clusters=3)
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

# ── Central registry ──────────────────────────────────────────────

_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}


# ── Registration ──────────────────────────────────────────────────

def register_model(name: str, *, defaults: dict | None = None, description="", paper="", uses_graphs=False,
                   implementation=""):
    """Class decorator that registers a model in the global registry.

    Parameters
    ----------
    name : str
        Unique lower-case key used to look up the model
        (e.g. ``"lmgec"``).
    defaults : dict, optional
        Default hyper-parameter values exposed to the CLI and
        config system.  If *None*, defaults are extracted
        automatically from the class ``__init__`` signature.

    Raises
    ------
    ValueError
        If *name* is already registered (prevents silent shadowing).

    Examples
    --------
    >>> @register_model("lmgec")
    ... class LMGEC(BaseModel): ...

    >>> @register_model("custom", defaults={"lr": 0.01, "epochs": 100})
    ... class CustomModel(BaseModel): ...
    """

    def decorator(cls: type) -> type:
        key = name.lower()
        if key in _MODEL_REGISTRY:
            raise ValueError(
                f"Model '{key}' is already registered "
                f"(class: {_MODEL_REGISTRY[key]['class'].__name__}). "
                f"Choose a different name."
            )

        _MODEL_REGISTRY[key] = {
            "class": cls,
            "defaults": defaults or _extract_defaults(cls),
            "description": description,
            "paper": paper,
            "implementation": implementation,
            "uses_graphs": uses_graphs,
            "supports_missing_views": getattr(cls, "supports_missing_views", False),
        }
        return cls

    return decorator


# ── Retrieval API ─────────────────────────────────────────────────

def get_model(name: str) -> type:
    """Return the model **class** registered under *name*.

    Parameters
    ----------
    name : str
        Case-insensitive model name.

    Raises
    ------
    ValueError
        If *name* is not found in the registry.
    """
    key = name.lower()
    if key not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY)) or "(none)"
        raise ValueError(
            f"Model '{name}' is not registered. "
            f"Available models: {available}"
        )
    return _MODEL_REGISTRY[key]["class"]


def get_model_info(name: str) -> dict:
    """Return the full registry entry for *name*.

    Returns a dict with keys ``"class"`` and ``"defaults"``.

    Raises
    ------
    ValueError
        If *name* is not found in the registry.
    """
    key = name.lower()
    if key not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY)) or "(none)"
        raise ValueError(
            f"Model '{name}' is not registered. "
            f"Available models: {available}"
        )
    return _MODEL_REGISTRY[key]


def list_models() -> list[str]:
    """Return a sorted list of all registered model names."""
    return sorted(_MODEL_REGISTRY.keys())


def list_models_info() -> dict[str, dict]:
    """Return the full registry dict (read-only copy)."""
    return dict(_MODEL_REGISTRY)


# ── Internal helpers ──────────────────────────────────────────────

def _extract_defaults(cls: type) -> dict:
    """Extract default parameter values from the class ``__init__``.

    Inspects the constructor signature and collects every parameter
    that has a default value, skipping ``self`` and ``n_clusters``
    (which is always dataset-dependent).
    """
    _SKIP = {"self", "n_clusters"}
    defaults = {}
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return defaults

    for name, param in sig.parameters.items():
        if name in _SKIP:
            continue
        if param.default is not inspect.Parameter.empty:
            defaults[name] = param.default

    return defaults
