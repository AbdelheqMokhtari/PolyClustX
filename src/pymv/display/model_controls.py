"""Typed model controls for the interface (no JSON editing required)."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from pymv.models import list_models
from pymv.models.registry import get_model_info


@dataclass(frozen=True)
class ParameterControl:
    label: str
    kind: str
    help: str = ""
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    search_values: tuple = ()
    strictly_positive: bool = False
    choices: tuple = ()
    nullable: bool = False


COMMON = {
    "theta": ParameterControl("Feature entropy weight", "float", minimum=0., strictly_positive=True,
                               search_values=(.1, 1., 4.)),
    "contrastive_weight": ParameterControl("Contrastive loss weight", "float", minimum=0.,
                                            search_values=(.1, 1., 5.)),
    "num_layers": ParameterControl("Filter depth", "int", minimum=1, search_values=(2, 5, 10)),
    "walks_per_node": ParameterControl("Walks per node", "int", minimum=1, search_values=(1, 2, 5)),
    "walk_length": ParameterControl("Walk length", "int", minimum=1, search_values=(3, 5, 10)),
    "context_size": ParameterControl("Walk context size", "int", minimum=2, search_values=(2, 3, 4)),
    "community_fraction": ParameterControl("Community positive fraction", "float", minimum=0., maximum=1.,
                                            search_values=(.1, .3, .5)),
    "filter_std": ParameterControl("Edge filter threshold (standard deviations)", "float", minimum=0.),
    "initial_fraction": ParameterControl("Initial curriculum fraction", "float", minimum=0., maximum=1.,
                                          strictly_positive=True, search_values=(.1, .25, .5)),
    "core_view": ParameterControl("Core view (zero-based)", "int", minimum=0),
    "augmentation_rate": ParameterControl("Single-view feature dropout", "float", minimum=0., maximum=.99,
                                           search_values=(.05, .1, .2)),
    "embedding_dim": ParameterControl("Embedding dimension", "int", minimum=1, step=1,
                                      search_values=(8, 16, 32, 64, 128, 256)),
    "projection_dim": ParameterControl("Projection dimension", "int", minimum=1, step=1,
                                       search_values=(32, 64, 128, 256)),
    "temperature": ParameterControl("Temperature", "float", minimum=0.0, strictly_positive=True, step=0.1,
                                    search_values=(0.1, 0.25, 0.5, 1.0, 2.0)),
    "beta": ParameterControl("Graph self-loop weight", "float", minimum=0.0, step=0.1,
                             search_values=(0.0, 0.5, 1.0, 2.0)),
    "tf_idf": ParameterControl("Use TF-IDF", "bool"),
    "center": ParameterControl("Center features", "bool"),
    "scale": ParameterControl("Scale features", "bool"),
    "preprocess": ParameterControl("Model-internal preprocessing", "bool"),
    "max_iter": ParameterControl("Training iterations", "int", minimum=1, step=1,
                                 search_values=(10, 30, 50, 100, 200)),
    "learning_rate": ParameterControl("Learning rate", "float", minimum=0.0, strictly_positive=True, step=0.0001,
                                      search_values=(0.0001, 0.0003, 0.001, 0.003)),
    "weight_decay": ParameterControl("Weight decay", "float", minimum=0.0, step=0.0001,
                                     search_values=(0.0, 0.0001, 0.001, 0.01)),
    "batch_size": ParameterControl("Batch size", "int", minimum=2, step=1,
                                   search_values=(64, 128, 256, 512)),
    "miss_rate": ParameterControl("Training view-mask rate", "float", minimum=0.0, maximum=1.0, step=0.05,
                                  search_values=(0.0, 0.1, 0.25, 0.4)),
    "noise_rate": ParameterControl("Training noise rate", "float", minimum=0.0, maximum=1.0, step=0.05,
                                   search_values=(0.0, 0.1, 0.25, 0.4)),
    "gaussian_noise": ParameterControl("Gaussian noise", "float", minimum=0.0, step=0.05,
                                       search_values=(0.0, 0.2, 0.4, 0.6)),
    "reconstruction_weight": ParameterControl("Reconstruction weight", "float", minimum=0.0,
                                              step=0.1, search_values=(0.0, 0.1, 0.5, 1.0)),
    "hidden_dims": ParameterControl("Hidden dimensions", "int_tuple",
                                    help="Comma-separated layer widths, for example 500, 500, 2000"),
    "prediction_dims": ParameterControl("Prediction network widths", "int_tuple",
                                         help="Comma-separated widths, for example 128, 256, 128"),
    "device": ParameterControl("Compute device", "choice", choices=("auto", "cpu", "cuda")),
    "affinity": ParameterControl("Affinity", "choice", choices=("rbf", "nearest_neighbors", "poly"),
                                  search_values=("rbf", "nearest_neighbors", "poly")),
    "init": ParameterControl("Initialization", "choice", choices=("k-means++", "random")),
    "n_init": ParameterControl("Initializations", "int", minimum=1, search_values=(1, 5, 10, 20)),
    "n_neighbors": ParameterControl("Nearest neighbors", "int", minimum=1, search_values=(5, 10, 20)),
    "patience": ParameterControl("Patience (sweeps)", "int", minimum=1),
    "tol": ParameterControl("Convergence tolerance", "float", minimum=0., step=.0001,
                             search_values=(0., .0001, .001)),
    "alpha": ParameterControl("Alpha", "float", minimum=0., strictly_positive=True, step=.1),
    "gamma": ParameterControl("Clustering loss weight", "float", minimum=0., step=.1,
                               search_values=(.01, .1, 1.)),
    "info_view": ParameterControl("Informative view (zero-based)", "int", minimum=0, nullable=True,
                                   help="Automatic concatenates embeddings from every view."),
    "degree": ParameterControl("Polynomial degree", "int", minimum=1),
    "coef0": ParameterControl("Polynomial offset", "float", minimum=0., step=.1),
    "max_memory_mb": ParameterControl("Memory safety budget (MiB)", "int", minimum=1,
                                       help="Preflight estimate, not a guarantee on total process memory."),
    "pretrain_epochs": ParameterControl("Reconstruction pretraining epochs", "int", minimum=1),
    "finetune_epochs": ParameterControl("Pseudo-label fine-tuning epochs", "int", minimum=1),
    "update_interval": ParameterControl("Target refresh interval (minibatches)", "int", minimum=1),
    "prediction_start": ParameterControl("Start prediction at epoch (zero-based)", "int", minimum=0,
                                          help="Must be below training epochs to learn missing-view completion."),
    "prediction_weight": ParameterControl("Prediction loss weight", "float", minimum=0., step=.1,
                                           search_values=(.01, .1, 1.)),
    "label_temperature": ParameterControl("Cluster contrast temperature", "float", minimum=0.,
                                           strictly_positive=True, step=.1, search_values=(.5, 1., 2.)),
}


def parse_candidates(values, control):
    """Validate user-entered numeric candidates without truncating integers."""
    if not values:
        raise ValueError(f"{control.label}: add at least one candidate")
    if control.kind == "choice":
        if any(value not in control.choices for value in values):
            raise ValueError(f"{control.label}: choose from {control.choices}")
        return list(dict.fromkeys(values))
    parsed = []
    for value in values:
        try:
            number = Decimal(str(value).strip())
        except InvalidOperation as exc:
            raise ValueError(f"{control.label}: {value!r} is not a number") from exc
        if not number.is_finite():
            raise ValueError(f"{control.label}: use finite numbers")
        if control.kind == "int" and number != number.to_integral_value():
            raise ValueError(f"{control.label}: use whole numbers")
        if control.strictly_positive and number <= 0:
            raise ValueError(f"{control.label}: values must be greater than zero")
        if control.minimum is not None and number < Decimal(str(control.minimum)):
            raise ValueError(f"{control.label}: values must be at least {control.minimum}")
        if control.maximum is not None and number > Decimal(str(control.maximum)):
            raise ValueError(f"{control.label}: values must be at most {control.maximum}")
        # Streamlit and downstream numerical libraries use finite double precision.
        floating = float(number)
        if not math.isfinite(floating) or (control.strictly_positive and floating == 0):
            raise ValueError(f"{control.label}: value is outside the supported numeric range")
        converted = int(number) if control.kind == "int" else floating
        if converted not in parsed:
            parsed.append(converted)
    return parsed


def parameter_controls(model_name):
    """Return ordered defaults and typed controls for a registered model."""
    info = get_model_info(model_name)
    controls = {}
    for name, default in info["defaults"].items():
        if name == "random_state":
            continue
        if name == "gamma" and model_name == "g_comvkm":
            controls[name] = ParameterControl("Collaboration exponent (greater than 1)", "float",
                                              minimum=1.000001, search_values=(2., 5., 10.))
        elif name == "gamma" and model_name == "sgla":
            controls[name] = ParameterControl("Laplacian weight regularization", "float", minimum=0.,
                                              search_values=(0., .1, .5, 1.))
        elif name == "gamma" and default is None:
            controls[name] = ParameterControl("Kernel scale (gamma)", "float", minimum=0.,
                                              strictly_positive=True, nullable=True,
                                              search_values=(.01, .1, 1.),
                                              help="Automatic estimates a separate scale for each view.")
        elif name == "max_iter" and model_name == "idec":
            controls[name] = ParameterControl("Clustering minibatch updates", "int", minimum=1,
                                              search_values=(100, 500, 1000))
        elif name == "reconstruction_weight" and model_name in {"completer", "dcp"}:
            controls[name] = ParameterControl("Reconstruction loss weight", "float", minimum=0.,
                                              strictly_positive=True, step=.1, search_values=(.01, .1, 1.))
        elif name in COMMON:
            controls[name] = COMMON[name]
        elif isinstance(default, bool):
            controls[name] = ParameterControl(name.replace("_", " ").title(), "bool")
        elif isinstance(default, int):
            controls[name] = ParameterControl(name.replace("_", " ").title(), "int", minimum=0, step=1)
        elif isinstance(default, float):
            controls[name] = ParameterControl(name.replace("_", " ").title(), "float", step=0.1)
        else:
            controls[name] = ParameterControl(name.replace("_", " ").title(), "text")
    return info["defaults"], controls


def compatible_models(data, preprocessing=None):
    """Split registered models into runnable and unavailable choices."""
    complete = data.is_complete if preprocessing is None else preprocessing.will_be_complete(data.is_complete)
    available, unavailable = [], []
    for name in list_models():
        info = get_model_info(name)
        min_complete = getattr(info["class"], "min_complete_samples", 0)
        # Missing simulation changes the mask, so validate the prepared data at run time too.
        too_few_pairs = (not complete and min_complete and data.observed_mask.all(axis=1).sum() < min_complete)
        if (complete or info.get("supports_missing_views", False)) and not too_few_pairs:
            available.append(name)
        else:
            unavailable.append(name)
    return available, unavailable


def constructor_accepts(model_name, parameter):
    return parameter in inspect.signature(get_model_info(model_name)["class"].__init__).parameters


def parse_int_tuple(value):
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("Layer widths must be comma-separated integers") from exc
    if not parsed or any(item < 1 for item in parsed):
        raise ValueError("Layer widths must be positive integers")
    return parsed
