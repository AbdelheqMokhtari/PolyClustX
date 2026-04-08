from .registry import (
    register_model,
    get_model,
    get_model_info,
    list_models,
    list_models_info,
)

# Import model sub-packages so they self-register via @register_model.
# Adding a new model only requires: create the class with the decorator
# and add a single import line here.
from .lmgec import LMGEC

__all__ = [
    # Registry API
    "register_model",
    "get_model",
    "get_model_info",
    "list_models",
    "list_models_info",
    # Models
    "LMGEC",
]
