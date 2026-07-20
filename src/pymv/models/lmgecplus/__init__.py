"""
Modular LMGEC research framework.

Public surface:

* :class:`LMGECplus` — the composable model (facade).
* Component interfaces + implementations (losses, normalizations,
  propagation, attention, weighting, alignment, clustering heads,
  optimizers) — each independently replaceable.
* :func:`register_component` / :func:`list_components` — extend the
  framework with one class, no library changes.
* :class:`LMGECResult` — structured fit output.
* :func:`build_ablation_suite` — one-call ablation studies.
"""

from .ablation import DEFAULT_ABLATIONS, build_ablation_suite
from .alignment import (
    Alignment,
    ConsensusAlignment,
    ContrastiveAlignment,
    CosineAlignment,
    NoAlignment,
)
from .attention import (
    Attention,
    NodeLevelAttention,
    StaticAttention,
    ViewLevelAttention,
)
from .clustering import ClusteringHead, GMMHead, HardKMeansHead, SoftKMeansHead
from .factory import (
    get_component,
    list_components,
    register_component,
    resolve_component,
)
from .interfaces import (
    ComponentBundle,
    FitState,
    GraphPropagation,
    Normalization,
    OptimizationInfo,
    Optimizer,
    ReconstructionLoss,
    ViewWeighting,
)
from .lmgecplus import LMGECplus
from .losses import HuberLoss, L2Loss, L21Loss
from .normalization import (
    L2Normalization,
    MinMaxNormalization,
    NoNormalization,
    NormalizationPipeline,
    TfIdfNormalization,
    WhiteningNormalization,
    ZScoreNormalization,
)
from .optimizer import AlternatingOptimizer
from .propagation import ControlledPropagation, NoPropagation, StandardPropagation
from .result import LMGECResult
from .weighting import (
    AttentionWeighting,
    EntropyWeighting,
    InertiaWeighting,
    UniformWeighting,
)

__all__ = [
    # model + results + ablation
    "LMGECplus",
    "LMGECResult",
    "build_ablation_suite",
    "DEFAULT_ABLATIONS",
    # component registry
    "register_component",
    "resolve_component",
    "get_component",
    "list_components",
    # interfaces
    "ReconstructionLoss",
    "Normalization",
    "GraphPropagation",
    "Attention",
    "ViewWeighting",
    "Alignment",
    "ClusteringHead",
    "Optimizer",
    "OptimizationInfo",
    "FitState",
    "ComponentBundle",
    # losses
    "L2Loss",
    "L21Loss",
    "HuberLoss",
    # normalizations
    "NoNormalization",
    "L2Normalization",
    "TfIdfNormalization",
    "ZScoreNormalization",
    "MinMaxNormalization",
    "WhiteningNormalization",
    "NormalizationPipeline",
    # propagation
    "NoPropagation",
    "StandardPropagation",
    "ControlledPropagation",
    # attention
    "NodeLevelAttention",
    "StaticAttention",
    "ViewLevelAttention",
    # weighting
    "UniformWeighting",
    "InertiaWeighting",
    "EntropyWeighting",
    "AttentionWeighting",
    # alignment
    "NoAlignment",
    "ConsensusAlignment",
    "CosineAlignment",
    "ContrastiveAlignment",
    # clustering heads
    "HardKMeansHead",
    "SoftKMeansHead",
    "GMMHead",
    # optimizer
    "AlternatingOptimizer",
]
