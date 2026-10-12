"""
LMGEC — Linear Multi-view Graph Embedding and Clustering.

"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from pymv.core import BaseModel
from pymv.models._torch_utils import resolve_device
from pymv.models.registry import register_model
from pymv.preprocessing import get_propagated_features

logger = logging.getLogger(__name__)


# update rules  F, W, And G


def _update_rule_F(XW, G, K):
    centers = XW.new_zeros((K, XW.shape[1]))
    centers.index_add_(0, G.long(), XW)
    counts = torch.bincount(G.long(), minlength=K).clamp_min(1)
    # Like TF unsorted_segment_mean, an empty cluster has a zero centroid.
    return centers / counts[:, None]


def _update_rule_W(X, F, G):
    U, _, Vh = torch.linalg.svd(X.T @ F[G.long()], full_matrices=False)
    return U @ Vh


def _update_rule_G(XW, F):
    # Chunk the direct squared-distance calculation to preserve TF's tie and
    # rounding behavior without an unbounded samples × clusters × dimensions tensor.
    chunk = max(1, 1_000_000 // max(F.numel(), 1))
    labels = []
    for start in range(0, len(XW), chunk):
        diff = XW[start:start + chunk, None, :] - F[None, :, :]
        labels.append(diff.square().sum(dim=-1).argmin(dim=1))
    return torch.cat(labels)


# method to initialise W, G, and F

def _max_embedding_dim(X) -> int:
    """Largest number of components TruncatedSVD can extract from *X*."""
    return min(X.shape) - 1


def _init_W(X, n_components, random_state=None):
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    svd.fit(X)
    return svd.components_.T


def _init_G_F(XW, k, random_state=None):
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    km.fit(XW)
    return km.labels_.astype(np.int32), km.cluster_centers_


@torch.no_grad()
def _train_loop(Xs, F, G, alphas, k, max_iter):
    """Original alternating updates and pre-update reconstruction loss, in float64."""
    losses = []
    for _ in range(max_iter):
        consensus = Xs[0].new_zeros((Xs[0].shape[0], F.shape[1]))
        loss = Xs[0].new_zeros(())
        for view, weight in zip(Xs, alphas):
            projection = _update_rule_W(view, F, G)
            consensus = consensus + weight * (view @ projection)
            reconstruction = (F @ projection.T)[G.long()]
            loss = loss + torch.linalg.vector_norm(view - reconstruction)
        G = _update_rule_G(consensus, F)
        F = _update_rule_F(consensus, G, k)
        losses.append(loss)
    return G, F, consensus, torch.stack(losses)


# Public model class

@register_model("lmgec", description="Linear multiview attributed graph embedding and clustering.",
                implementation="https://github.com/chakib401/LMGEC",
                paper="https://doi.org/10.1145/3539597.3570367", uses_graphs=True, defaults={
    "embedding_dim": 16,
    "temperature": 1.0,
    "beta": 1.0,
    "tf_idf": False,
    "center": False,
    "scale": False,
    "preprocess": True,
    "max_iter": 30,
    "random_state": None,
    "device": "auto",
})
class LMGEC(BaseModel):
    """Linear Multi-view Graph Embedding Clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters *k*.

    embedding_dim : int
        Latent dimension *m* for the per-view projections W.

    temperature : float, default=1.0
        Softmax temperature for computing per-view weights α.
        Lower → sharper weighting; higher → more uniform.

    beta : float or list[float], default=1.0
        Self-loop weight for graph propagation.  A single float
        applies the same value to every view.  A list sets a
        per-view value.  **Ignored when ``preprocess=False``.**

    tf_idf : bool or list[bool], default=False
        Feature normalization mode.  ``True`` → TF-IDF + L2;
        ``False`` → plain L2.  A list sets a per-view choice.
        **Ignored when ``preprocess=False``.**

    center : bool, default=False
        Center features (subtract column mean) after propagation.
        **Ignored when ``preprocess=False``.**

    scale : bool, default=False
        Scale features (unit variance) after propagation.
        **Ignored when ``preprocess=False``.**

    preprocess : bool, default=True
        Master switch for internal preprocessing.

        * ``True``  — the model normalizes features, builds the
          propagation matrix, computes H = S·X, and optionally
          centers/scales.  This is the default and matches the
          original LMGEC paper.
        * ``False`` — **all** internal preprocessing is skipped.
          The model treats each element of *Xs* as an
          already-prepared view and feeds it directly into the
          optimisation loop.  In this mode *As* is ignored even
          if provided, and *beta*, *tf_idf*, *center*, *scale*
          have no effect.

    max_iter : int, default=30
        Maximum number of alternating-optimisation iterations.

    random_state : int or None, default=None
        Seed for the randomized initialisation (TruncatedSVD for W,
        KMeans for G/F).  Set it to make a fit reproducible; leave it
        as *None* for a different initialisation on every call.

    device : str, default="auto"
        ``auto`` selects CUDA when available and retries on CPU after CUDA
        out-of-memory. ``cpu`` and ``cuda[:index]`` select an explicit device.
        Results are NumPy arrays on CPU; optimization uses float64.

    Examples
    --------
    **Let the model preprocess (default):**

    >>> model = LMGEC(n_clusters=3, embedding_dim=4)
    >>> model.fit(Xs, As)          # raw features + adjacencies
    >>> model.labels_
    array([0, 2, 1, ...])

    **Per-view preprocessing options:**

    >>> model = LMGEC(
    ...     n_clusters=3,
    ...     embedding_dim=4,
    ...     tf_idf=[True, False],  # TF-IDF on view 0, L2 on view 1
    ...     beta=[1.0, 0.5],       # different self-loop weights
    ... )
    >>> model.fit(Xs, As)

    **User handles preprocessing externally:**

    >>> from pymv.preprocessing import get_propagated_features
    >>> H0 = get_propagated_features(X0, A0, beta=1.0, tf_idf=True)
    >>> H1 = get_propagated_features(X1, A1, beta=0.5, tf_idf=False)
    >>> model = LMGEC(n_clusters=3, embedding_dim=4, preprocess=False)
    >>> model.fit([H0, H1])        # As not needed

    **No graph at all (feature-only views):**

    >>> model = LMGEC(n_clusters=3, embedding_dim=4, preprocess=False)
    >>> model.fit([X1, X2])        # just feature matrices
    """

    @staticmethod
    def resolve_fit_parameters(Xs, params):
        """Return the effective parameters for these views, without modifying inputs."""
        resolved = dict(params)
        limit = min(min(view.shape) - 1 for view in Xs)
        if limit < 1:
            raise ValueError("LMGEC needs at least two samples and two features in every view")
        if "embedding_dim" in resolved:
            resolved["embedding_dim"] = min(resolved["embedding_dim"], limit)
        return resolved

    def __init__(
        self,
        n_clusters: int,
        embedding_dim: int,
        temperature: float = 1.0,
        beta: float | list[float] = 1.0,
        tf_idf: bool | list[bool] = False,
        center: bool = False,
        scale: bool = False,
        preprocess: bool = True,
        max_iter: int = 30,
        random_state: int | None = None,
        device: str = "auto",
    ):
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.beta = beta
        self.tf_idf = tf_idf
        self.center = center
        self.scale = scale
        self.preprocess = preprocess
        self.max_iter = max_iter
        self.random_state = random_state
        self.device = device

        # Filled after fit()
        self.embedding_dim_ = None
        self.labels_ = None
        self.centroids_ = None
        self.embedding_ = None
        self.loss_history_ = None
        self.alphas_ = None

    # helpers

    @property
    def name(self) -> str:
        return getattr(self, "_name", self.__class__.__name__)

    @name.setter
    def name(self, value: str):
        self._name = value

    def _resolve_per_view(self, param, n_views, param_name: str):
        """Broadcast a scalar parameter to a per-view list."""
        if isinstance(param, list):
            if len(param) != n_views:
                raise ValueError(
                    f"len({param_name})={len(param)} but got "
                    f"{n_views} views."
                )
            return param
        return [param] * n_views

    # main entry point

    def fit(self, Xs, As=None, *, observed_mask=None):
        """Fit the model.

        Parameters
        ----------
        Xs : list[np.ndarray]
            Feature matrices — one per view.

            * When ``preprocess=True``: raw features; each will be
              normalized and (if *As* is given) propagated through
              its adjacency.
            * When ``preprocess=False``: already-prepared views
              that go directly into the optimisation loop.

        As : list[np.ndarray | sp.spmatrix] or None, default=None
            One entry per view: adjacency, relation list, or None.

            * Ignored entirely when ``preprocess=False``.
            * When ``preprocess=True`` and ``As is None``: features
              are normalized but **not** propagated (no graph).
            * Shared graphs must be repeated explicitly. Relation lists
              are summed before propagation; for custom weighting or
              Hadamard fusion, use ``data.fit(model, fusion=..., weights=...)``.

        Returns
        -------
        self
            The fitted model with ``labels_``, ``centroids_``,
            ``embedding_``, ``loss_history_`` and ``embedding_dim_``
            attributes set.  ``embedding_dim_`` is the dimension
            actually used, which is capped by the rank of the
            lowest-rank view.
        """

        if observed_mask is not None and not np.asarray(observed_mask, dtype=bool).all():
            raise ValueError("LMGEC does not support missing views; explicitly impute first")
        n_views = len(Xs)
        if not n_views:
            raise ValueError("At least one view is required")
        if As is not None and len(As) != n_views:
            raise ValueError("Provide exactly one adjacency entry per view; explicitly repeat shared graphs")
        for name in ("n_clusters", "embedding_dim", "max_iter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if any(len(x.shape) != 2 or x.shape[0] != Xs[0].shape[0] for x in Xs):
            raise ValueError("All feature views must be two-dimensional and share the same sample count")
        if self.n_clusters > Xs[0].shape[0]:
            raise ValueError("n_clusters cannot exceed the number of samples")
        self.resolve_fit_parameters(Xs, {"embedding_dim": self.embedding_dim})
        device = resolve_device(self.device)

        # Prepare H views

        if self.preprocess:
            betas = self._resolve_per_view(self.beta, n_views, "beta")
            tf_idfs = self._resolve_per_view(self.tf_idf, n_views, "tf_idf")

            # Build adjacency list
            if As is None:
                adj_list = [None] * n_views
            else:
                adj_list = list(As)

            # Per-view pipeline (correct order):
            #   Raw → center/scale (column-wise) → L2 normalize (row-wise) → propagate
            # Centering and scaling must precede row normalization so that
            # feature distributions are aligned before projecting onto the
            # unit sphere.
            Hs = []
            for v in range(n_views):
                H = get_propagated_features(
                    Xs[v],
                    adj=adj_list[v],
                    beta=betas[v],
                    tf_idf=tf_idfs[v],
                    center=self.center,
                    scale=self.scale,
                )
                Hs.append(H)

        else:
            # User already prepared the views — use as-is
            Hs = list(Xs)

        # Ensure dense float64 arrays

        for i in range(len(Hs)):
            if sp.issparse(Hs[i]):
                Hs[i] = Hs[i].toarray()
            Hs[i] = np.asarray(Hs[i], dtype=np.float64)
            if not np.isfinite(Hs[i]).all():
                raise ValueError("Xs contains NaN or infinite values")

        # Initialisation

        # Every view must project into the *same* latent space, so the
        # embedding dimension is capped by the lowest-rank view — not
        # per view, which would make the consensus shapes disagree.
        max_dim = min(_max_embedding_dim(H) for H in Hs)
        m = min(self.embedding_dim, max_dim)
        if m < self.embedding_dim:
            logger.warning(
                "embedding_dim=%d exceeds the rank limit of the smallest "
                "view — using %d components instead.",
                self.embedding_dim, m,
            )
        self.embedding_dim_ = m

        k = self.n_clusters

        alphas = np.zeros(n_views, dtype=np.float64)
        self.per_view_results_ = []
        XW_consensus = np.zeros((Hs[0].shape[0], m), dtype=np.float64)

        # Inertia is normalized by sqrt(N) so the softmax scale is invariant
        # to dataset size.  Without it, large N makes the exponential collapse
        # onto a single view and the model stops being multi-view.
        for v in range(n_views):
            Wv = _init_W(Hs[v], m, random_state=self.random_state)
            XWv = Hs[v] @ Wv
            Gv, Fv = _init_G_F(XWv, k, random_state=self.random_state)
            inertia = np.linalg.norm(XWv - Fv[Gv]) / np.sqrt(XWv.shape[0])
            self.per_view_results_.append({"view": v, "stage": "initialization",
                                           "normalized_inertia": float(inertia)})
            logger.info("[View %d] normalized inertia: %.4f", v, inertia)

            alphas[v] = np.exp(-inertia / self.temperature)
            XW_consensus += alphas[v] * XWv

        alpha_sum = alphas.sum()
        if alpha_sum > 0:
            XW_consensus /= alpha_sum
        else:
            XW_consensus /= n_views

        G_init, F_init = _init_G_F(XW_consensus, k, random_state=self.random_state)

        # run training loop (with automatic GPU → CPU fallback on OOM)

        alphas_norm = alphas / (alpha_sum if alpha_sum > 0 else n_views)

        def _run_on_device(target):
            """Create tensors and run the loop on the given device."""
            _Hs = tuple(torch.as_tensor(H, dtype=torch.float64, device=target) for H in Hs)
            _F = torch.as_tensor(F_init, dtype=torch.float64, device=target)
            _G = torch.as_tensor(G_init, dtype=torch.long, device=target)
            _a = torch.as_tensor(alphas_norm, dtype=torch.float64, device=target)
            return tuple(t.cpu().numpy() for t in _train_loop(_Hs, _F, _G, _a, k, self.max_iter))

        output = None
        try:
            output = _run_on_device(device)
        except torch.cuda.OutOfMemoryError:
            if self.device != "auto" or device.type != "cuda":
                raise
            logger.warning("CUDA out of memory; retrying LMGEC on CPU")
        if output is None:
            # Retry outside the exception block so its traceback releases GPU tensors.
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            output = _run_on_device(device)
        G_out, F_out, XW_out, losses = output
        self.device_ = str(device)

        # Store results & free GPU memory

        self.labels_ = G_out.astype(np.int32, copy=False)
        self.centroids_ = F_out
        self.embedding_ = XW_out
        self.loss_history_ = losses
        self.alphas_ = alphas_norm
        residual = self.embedding_ - self.centroids_[self.labels_]
        self.inertia_ = float(np.einsum("ij,ij->", residual, residual))

        return self

    def fit_predict(self, Xs, As=None):
        """Fit the model and return cluster labels.

        Parameters
        ----------
        Xs : list[np.ndarray]
            Feature matrices (see :meth:`fit`).

        As : list or None, default=None
            Adjacency matrices (see :meth:`fit`).

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Cluster label for each sample.
        """
        return self.fit(Xs, As).labels_
