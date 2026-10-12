"""Serializable preprocessing recipes used by the interactive interface."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from polyclustx.data import impute_missing, simulate_missing_views
from polyclustx.preprocessing import get_propagated_features, preprocess_features


@dataclass
class ViewPreprocessing:
    """Operations applied to one feature view before fitting a model."""

    normalize: bool = False
    tf_idf: bool = False
    center: bool = False
    scale: bool = False
    propagate: bool = False
    beta: float = 1.0


@dataclass
class PreprocessingSpec:
    """Complete, reusable data preparation recipe for one experiment."""

    mode: str = "none"
    selected_views: list[int] = field(default_factory=list)
    shared: ViewPreprocessing = field(default_factory=ViewPreprocessing)
    per_view: dict[int, ViewPreprocessing] = field(default_factory=dict)
    missing_rate: float = 0.0
    missing_seed: int = 0
    imputation: str | None = None
    fusion: str = "sum"
    relation_weights: dict[int, list[float]] = field(default_factory=dict)

    def settings_for(self, view: int) -> ViewPreprocessing | None:
        if self.mode == "none":
            return None
        if self.mode == "all":
            return self.shared
        if self.mode == "selected":
            return self.shared if view in self.selected_views else None
        if self.mode == "per_view":
            return self.per_view.get(view)
        raise ValueError("Preprocessing mode must be none, all, selected, or per_view")

    @property
    def changes_features(self) -> bool:
        configurations = [self.shared, *self.per_view.values()]
        return any(
            value.normalize or value.tf_idf or value.center or value.scale or value.propagate
            for value in configurations
        ) and self.mode != "none"

    def will_be_complete(self, currently_complete: bool) -> bool:
        return (currently_complete and self.missing_rate == 0) or self.imputation is not None


def prepare_dataset(source, recipe: PreprocessingSpec):
    """Copy and prepare a dataset without modifying the session resource."""
    if not 0 <= recipe.missing_rate < 1:
        raise ValueError("Missing-view fraction must be in [0, 1)")
    if recipe.imputation not in {None, "mean", "zero"}:
        raise ValueError("Imputation must be None, 'mean', or 'zero'")

    data = (
        simulate_missing_views(source, recipe.missing_rate, random_state=recipe.missing_seed)
        if recipe.missing_rate
        else copy.deepcopy(source)
    )
    if recipe.imputation:
        data = impute_missing(data, recipe.imputation)

    weights = {int(view): values for view, values in recipe.relation_weights.items()}
    data.As = data.fused_graphs(recipe.fusion, weights)

    if not data.is_complete and recipe.changes_features:
        raise ValueError("Feature preprocessing needs complete views. Select mean or zero imputation first.")

    for view, features in enumerate(data.Xs):
        settings = recipe.settings_for(view)
        if settings is None:
            continue
        graph = data.As[view] if settings.propagate else None
        if settings.normalize or settings.tf_idf or settings.propagate:
            data.Xs[view] = get_propagated_features(
                features,
                graph,
                beta=settings.beta,
                tf_idf=settings.tf_idf,
                center=settings.center,
                scale=settings.scale,
            )
        elif settings.center or settings.scale:
            data.Xs[view] = preprocess_features(features, center=settings.center, scale=settings.scale)

    data.metadata["display_preprocessing"] = {
        "mode": recipe.mode,
        "selected_views": list(recipe.selected_views),
        "missing_rate": recipe.missing_rate,
        "imputation": recipe.imputation,
        "fusion": recipe.fusion,
    }
    return data
