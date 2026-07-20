"""
Component registry + resolution for the modular LMGEC framework.

Mirrors the model-level registry (:mod:`pymv.models.registry`) one
level down: every component *kind* (loss, normalization, propagation,
attention, weighting, alignment, clustering, optimizer) has its own
namespace of named implementations.

Adding a new component requires **one class + one decorator**::

    from pymv.models.lmgecplus import ReconstructionLoss, register_component

    @register_component("loss", "cauchy")
    class CauchyLoss(ReconstructionLoss):
        ...

    LMGECplus(n_clusters=3, embedding_dim=8, reconstruction_loss="cauchy")

Specs accepted by :func:`resolve_component`
-------------------------------------------
* ``None``            → the caller-supplied default
* an instance         → used as-is
* a class             → instantiated with no arguments
* a string            → registry lookup (``"l21"``, ``"soft"``, …)
* a dict              → ``{"name": <str>, <ctor kwargs>...}``; the keys
  ``"method"`` and ``"type"`` are accepted as aliases of ``"name"`` so
  YAML configs read naturally (``clustering: {method: soft}``).
  ``{"enabled": false}`` resolves to the kind's ``"none"`` component.
"""

from __future__ import annotations

from typing import Any, Dict

COMPONENT_KINDS = (
    "loss", "normalization", "propagation", "attention",
    "weighting", "alignment", "clustering", "optimizer",
)

_COMPONENT_REGISTRY: Dict[str, Dict[str, type]] = {k: {} for k in COMPONENT_KINDS}

#: dict-spec name used when ``{"enabled": true}`` is given without an
#: explicit name (e.g. ``graph_propagation: {enabled: true, gamma: 0.4}``).
_DEFAULT_DICT_NAME = {
    "propagation": "controlled",
    "alignment": "consensus",
    "attention": "dynamic",
}


def register_component(kind: str, name: str, *, aliases: tuple = ()):
    """Class decorator registering a component under ``kind``/``name``.

    Parameters
    ----------
    kind : str
        One of :data:`COMPONENT_KINDS`.
    name : str
        Unique lower-case key inside that kind.
    aliases : tuple[str], optional
        Additional lookup names for the same class (e.g. the original
        LMGEC weighting is reachable as both ``"inertia"`` and
        ``"original"``).
    """
    if kind not in _COMPONENT_REGISTRY:
        raise ValueError(
            f"Unknown component kind '{kind}'. "
            f"Kinds: {', '.join(COMPONENT_KINDS)}"
        )

    def decorator(cls: type) -> type:
        for key in (name, *aliases):
            key = key.lower()
            if key in _COMPONENT_REGISTRY[kind]:
                raise ValueError(
                    f"Component '{key}' already registered for kind "
                    f"'{kind}' (class: "
                    f"{_COMPONENT_REGISTRY[kind][key].__name__})."
                )
            _COMPONENT_REGISTRY[kind][key] = cls
        return cls

    return decorator


def list_components(kind: str | None = None):
    """List registered component names (all kinds, or one kind)."""
    if kind is not None:
        return sorted(_COMPONENT_REGISTRY[kind].keys())
    return {k: sorted(v.keys()) for k, v in _COMPONENT_REGISTRY.items()}


def get_component(kind: str, name: str) -> type:
    """Return the class registered under ``kind``/``name``."""
    try:
        return _COMPONENT_REGISTRY[kind][name.lower()]
    except KeyError:
        available = ", ".join(sorted(_COMPONENT_REGISTRY.get(kind, {}))) or "(none)"
        raise ValueError(
            f"Unknown {kind} component '{name}'. Available: {available}"
        ) from None


def resolve_component(kind: str, spec: Any, default: Any = None, **overrides):
    """Turn a user/config *spec* into a component instance.

    Parameters
    ----------
    kind : str
        Component kind (namespace for string lookups).
    spec : None | str | dict | class | instance
        See module docstring for the accepted forms.
    default : Any
        Returned (after resolution) when ``spec is None``.  May itself
        be any accepted spec form.
    **overrides
        Extra constructor kwargs applied when *spec* is a string, class
        or dict (ignored for ready-made instances).
    """
    if spec is None:
        if default is None:
            raise ValueError(f"No spec and no default for kind '{kind}'.")
        spec = default

    # Ready-made instance → trust the caller.
    if not isinstance(spec, (str, dict, type)):
        return spec

    if isinstance(spec, type):
        return spec(**overrides)

    if isinstance(spec, dict):
        params = dict(spec)
        name = None
        for key in ("name", "method", "type"):
            if key in params:
                name = str(params.pop(key))
                break
        if "enabled" in params:
            enabled = params.pop("enabled")
            if not enabled:
                name = "none"
            elif name is None:
                name = _DEFAULT_DICT_NAME.get(kind)
        if name is None:
            raise ValueError(
                f"Dict spec for '{kind}' needs a 'name'/'method' key "
                f"(got keys: {sorted(spec)})."
            )
        params.update(overrides)
        return get_component(kind, name)(**params)

    # plain string
    return get_component(kind, spec)(**overrides)
