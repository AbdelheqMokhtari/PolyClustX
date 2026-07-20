"""
Ready-made ablation suites for the modular LMGEC framework.

The default ladder follows the **eight LMGEC extensions** (see
``docs/LMGEC_extensions.md`` for the maths of each one): the exact
baseline (original LMGEC, every extension off) plus one variant per
extension, each **added alone**, and one *full* model combining them.
Because every component of :class:`~pymv.models.lmgecplus.LMGECplus`
is independently pluggable, a variant is just a dictionary of keyword
overrides — no code duplication, no model subclasses.

=====  =============================  ===================================
 #     Extension                      Override
=====  =============================  ===================================
 E1    robust ℓ2,1 loss               ``reconstruction_loss="l21"``
 E2    node-level view attention      ``attention="static"``
 E3    dim-normalized errors          ``attention={"name": "static",
                                      "dim_normalized": True}``
 E4    dynamic α per iteration        ``attention="view"``
 E5    learned regularisation         ``attention={"name": "node",
       (entropy / l2)                 "regularization": "entropy"}``
 E6    controlled propagation γ       ``graph_propagation={"enabled":
                                      True, "gamma": 0.5}``
 E7    soft assignments               ``clustering="soft"``
 E8    cross-view alignment           ``alignment="consensus"``
=====  =============================  ===================================

E3 and E5 necessarily *build on* attention (without attention the
error normalisation / regularised allocation has nothing to weight);
they isolate the marginal effect of the normalisation / regulariser
on top of E2.

Example
-------
>>> from pymv.experiment import Experiment
>>> from pymv.metrics import clustering_accuracy, nmi_score
>>> from pymv.models.lmgecplus import build_ablation_suite
>>>
>>> models = build_ablation_suite(n_clusters=3, embedding_dim=8)
>>> Experiment(datasets=["acm"], models=list(models.values()),
...            metrics=[clustering_accuracy, nmi_score], runs=3).run()
"""

from __future__ import annotations

from .lmgecplus import LMGECplus

#: name → constructor overrides: baseline, each extension alone, full.
DEFAULT_ABLATIONS: dict[str, dict] = {
    "baseline": {},
    "E1 robust-l21": {
        "reconstruction_loss": "l21",
    },
    "E2 node-attention": {
        "attention": "static",
    },
    "E3 attn+dim-norm": {
        "attention": {"name": "static", "dim_normalized": True},
    },
    "E4 dynamic-alphas": {
        "attention": "view",
    },
    "E5 attn+entropy-reg": {
        "attention": {"name": "node", "regularization": "entropy"},
    },
    "E6 controlled-prop": {
        "graph_propagation": {"enabled": True, "gamma": 0.5},
    },
    "E7 soft-clustering": {
        "clustering": "soft",
    },
    "E8 alignment": {
        "alignment": "consensus",
    },
    "full": {
        "reconstruction_loss": "l21",
        "attention": {"name": "node", "dim_normalized": True,
                      "regularization": "entropy"},
        "graph_propagation": {"enabled": True, "gamma": 0.5},
        "clustering": "soft",
        "alignment": "consensus",
    },
}


def build_ablation_suite(
    n_clusters: int,
    embedding_dim: int = 16,
    variants: dict[str, dict] | None = None,
    include: list[str] | None = None,
    **base_kwargs,
) -> dict[str, LMGECplus]:
    """Build one named :class:`LMGECplus` per ablation variant.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (shared by all variants).
    embedding_dim : int, default=16
        Shared latent dimension.
    variants : dict[str, dict] or None
        ``{variant name: constructor overrides}``.  ``None`` → the
        standard eight-extension ladder in :data:`DEFAULT_ABLATIONS`
        (baseline, E1–E8 each added alone, full model).
    include : list[str] or None
        Restrict the suite to these variant names (``None`` → all).
    **base_kwargs
        Shared constructor arguments applied to every variant
        (``max_iter``, ``random_state``, ``temperature``, …).
        Variant overrides win on conflict.

    Returns
    -------
    dict[str, LMGECplus]
        Ordered mapping of variant name → configured (unfitted) model.
        Each model's ``name`` is set so experiment tables read well.
    """
    variants = DEFAULT_ABLATIONS if variants is None else variants
    if include is not None:
        unknown = set(include) - set(variants)
        if unknown:
            raise ValueError(
                f"Unknown ablation variant(s): {sorted(unknown)}. "
                f"Available: {sorted(variants)}"
            )
        variants = {k: variants[k] for k in include}

    suite: dict[str, LMGECplus] = {}
    for name, overrides in variants.items():
        kwargs = {**base_kwargs, **overrides}
        model = LMGECplus(
            n_clusters=n_clusters, embedding_dim=embedding_dim, **kwargs)
        model.name = name
        suite[name] = model
    return suite
