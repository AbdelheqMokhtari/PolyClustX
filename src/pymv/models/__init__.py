# Import model sub-packages so they self-register via @register_model.
# Adding a new model only requires: create the class with the decorator
# and add a single import line here.
from .completer import COMPLETER, Completer
from .conan import CONAN
from .dcp import DCP
from .dgf import DGF
from .g_comvkm import GCoMVKM
from .idec import IDEC
from .lmgec import LMGEC
from .mflvc import MFLVC
from .multiview_kmeans import MultiViewKMeans, MultiviewKMeans
from .multiview_spectral import MultiViewSpectralClustering, MultiviewSpectralClustering
from .registry import get_model, get_model_info, list_models, list_models_info, register_model
from .scm import SCM
from .sgla import SGLA
from .umc import UMC

__all__ = [
    "register_model",
    "get_model",
    "get_model_info",
    "list_models",
    "list_models_info",
    "LMGEC",
    "SCM",
    "CONAN",
    "SGLA",
    "UMC",
    "DGF",
    "GCoMVKM",
    "IDEC",
    "COMPLETER",
    "Completer",
    "DCP",
    "MFLVC",
    "MultiViewKMeans",
    "MultiviewKMeans",
    "MultiViewSpectralClustering",
    "MultiviewSpectralClustering",
]
