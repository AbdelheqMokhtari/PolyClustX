"""Typed model controls for the interface (no JSON editing required)."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

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


COMMON = {
    "embedding_dim": ParameterControl("Embedding dimension", "int", minimum=2, maximum=4096, step=1,
                                      search_values=(8, 16, 32, 64, 128, 256)),
    "projection_dim": ParameterControl("Projection dimension", "int", minimum=2, maximum=4096, step=1,
                                       search_values=(32, 64, 128, 256)),
    "temperature": ParameterControl("Temperature", "float", minimum=0.001, maximum=100.0, step=0.1,
                                    search_values=(0.1, 0.25, 0.5, 1.0, 2.0)),
    "beta": ParameterControl("Graph self-loop weight", "float", minimum=0.0, maximum=100.0, step=0.1,
                             search_values=(0.0, 0.5, 1.0, 2.0)),
    "tf_idf": ParameterControl("Use TF-IDF", "bool"),
    "center": ParameterControl("Center features", "bool"),
    "scale": ParameterControl("Scale features", "bool"),
    "preprocess": ParameterControl("Model-internal preprocessing", "bool"),
    "max_iter": ParameterControl("Training iterations", "int", minimum=1, maximum=100000, step=1,
                                 search_values=(10, 30, 50, 100, 200)),
    "learning_rate": ParameterControl("Learning rate", "float", minimum=1e-7, maximum=1.0, step=0.0001,
                                      search_values=(0.0001, 0.0003, 0.001, 0.003)),
    "weight_decay": ParameterControl("Weight decay", "float", minimum=0.0, maximum=1.0, step=0.0001,
                                     search_values=(0.0, 0.0001, 0.001, 0.01)),
    "batch_size": ParameterControl("Batch size", "int", minimum=2, maximum=100000, step=1,
                                   search_values=(64, 128, 256, 512)),
    "miss_rate": ParameterControl("Training view-mask rate", "float", minimum=0.0, maximum=1.0, step=0.05,
                                  search_values=(0.0, 0.1, 0.25, 0.4)),
    "noise_rate": ParameterControl("Training noise rate", "float", minimum=0.0, maximum=1.0, step=0.05,
                                   search_values=(0.0, 0.1, 0.25, 0.4)),
    "gaussian_noise": ParameterControl("Gaussian noise", "float", minimum=0.0, maximum=10.0, step=0.05,
                                       search_values=(0.0, 0.2, 0.4, 0.6)),
    "reconstruction_weight": ParameterControl("Reconstruction weight", "float", minimum=0.0, maximum=100.0,
                                              step=0.1, search_values=(0.0, 0.1, 0.5, 1.0)),
    "hidden_dims": ParameterControl("Hidden dimensions", "int_tuple",
                                    help="Comma-separated layer widths, for example 500, 500, 2000"),
}


def parameter_controls(model_name):
    """Return ordered defaults and typed controls for a registered model."""
    info = get_model_info(model_name)
    controls = {}
    for name, default in info["defaults"].items():
        if name == "random_state":
            continue
        if name in COMMON:
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
        if complete or info.get("supports_missing_views", False):
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
