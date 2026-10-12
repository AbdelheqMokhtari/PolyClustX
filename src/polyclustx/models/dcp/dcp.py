"""DCP's unsupervised clustering route (no label-based classification loss)."""
from ..completer.completer import _PredictionClustering
from ..registry import register_model


@register_model("dcp", description="Dual contrastive prediction, complete-graph clustering route for incomplete views.",
                paper="https://doi.org/10.1109/TPAMI.2022.3197238",
                implementation="https://github.com/XLearning-SCU/2022-TPAMI-DCP")
class DCP(_PredictionClustering):
    """DCP clustering with complete-graph prediction for two or more views.

    Uses the reference's instance-consistency marginal weight ``alpha``
    (COMPLETER uses ``alpha + 1``), reconstruction and delayed dual prediction.
    For >2 views, retains the reference complete-graph pair weighting. Category
    supervision, classification and core-view variants are not part of this
    clustering estimator. One view reduces to autoencoder + K-means.
    """

    _dcp = True

    def __init__(self, n_clusters, *, embedding_dim=128, hidden_dims=(256, 256),
                 prediction_dims=(128, 256, 128), batchnorm=True, alpha=10.,
                 reconstruction_weight=.1, prediction_weight=.1, prediction_start=10,
                 max_iter=100, learning_rate=.0001, batch_size=256, n_init=10,
                 preprocess=True, random_state=None, device="auto", max_memory_mb=1024):
        super().__init__(n_clusters, embedding_dim=embedding_dim, hidden_dims=hidden_dims,
                         prediction_dims=prediction_dims, batchnorm=batchnorm, alpha=alpha,
                         reconstruction_weight=reconstruction_weight, prediction_weight=prediction_weight,
                         prediction_start=prediction_start, max_iter=max_iter, learning_rate=learning_rate,
                         batch_size=batch_size, n_init=n_init, preprocess=preprocess, random_state=random_state,
                         device=device, max_memory_mb=max_memory_mb)
