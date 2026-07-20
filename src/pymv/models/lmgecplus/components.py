"""
Modular building blocks for :class:`~pymv.models.lmgecplus.LMGECplus`.

Every extension proposed for LMGECplus is implemented here as a small,
**independent**, side-effect-free NumPy function so that it can be
enabled, disabled, combined, and unit-tested in isolation.  The main
``LMGECplus`` class simply wires these together according to its
boolean flags.

The functions fall into five groups:

* **Projection / clustering primitives** (shared with LMGEC):
  :func:`procrustes_W`, :func:`hard_assign`, :func:`centroids_from_Z`.
* **Weighting** (extensions 2-5):
  :func:`reconstruction_error`, :func:`view_attention`,
  :func:`simplex_project`.
* **Robust loss** (extension 1): :func:`irls_weights`.
* **Graph propagation** (extension 6):
  :func:`controlled_propagation`, :func:`graph_smoothness`.
* **Soft clustering** (extension 7): :func:`soft_assign`.

Cross-view alignment (extension 8) needs no helper of its own — it is a
one-line target augmentation inside the W-update (see ``LMGECplus``).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

EPS = 1e-12


# Projection / clustering primitives (identical maths to LMGEC)

def procrustes_W(H, T, node_w=None):
    r"""Orthogonal Procrustes projection.

    Solves ``min_W ||diag(w)^{1/2} (H - T Wᵀ)||_F²`` s.t. ``WᵀW = I``
    via the SVD of ``Hᵀ diag(w) T``.  With ``node_w=None`` (all ones)
    this is exactly LMGEC's ``W = U Vᵀ`` from ``svd(Hᵀ T)``.

    Parameters
    ----------
    H : (n, d) array — view features.
    T : (n, m) array — reconstruction target (``F[G]`` or ``G_soft F``,
        optionally augmented for cross-view alignment).
    node_w : (n,) array or None — per-node weights (attention × robust).

    Returns
    -------
    W : (d, m) orthonormal-column projection.
    """
    if node_w is None:
        M = H.T @ T
    else:
        M = H.T @ (node_w[:, None] * T)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def hard_assign(Z, F):
    """Nearest-centroid hard assignment (LMGEC's ``argmin``)."""
    # ||z_i - f_c||² = ||z||² - 2 z·f + ||f||²  → argmin over c
    d2 = (
        (Z ** 2).sum(1)[:, None]
        - 2.0 * Z @ F.T
        + (F ** 2).sum(1)[None, :]
    )
    return np.argmin(d2, axis=1).astype(np.int32)


def centroids_from_Z(Z, R, k, node_w=None):
    """Centroid update from the consensus embedding.

    Accepts either hard labels (``R`` is an int vector) or a soft
    responsibility matrix (``R`` is ``(n, k)``).  With hard labels and
    ``node_w=None`` this is LMGEC's per-cluster mean.

    Parameters
    ----------
    Z : (n, m) consensus embedding.
    R : (n,) int labels  **or**  (n, k) responsibilities.
    k : int — number of clusters.
    node_w : (n,) or None — optional per-node weights (robust mean).
    """
    if R.ndim == 1:  # hard labels → one-hot
        Q = np.zeros((Z.shape[0], k), dtype=Z.dtype)
        Q[np.arange(Z.shape[0]), R] = 1.0
    else:
        Q = R
    if node_w is not None:
        Q = Q * node_w[:, None]
    num = Q.T @ Z                       # (k, m)
    den = Q.sum(0)[:, None]            # (k, 1)
    den = np.maximum(den, EPS)
    return num / den


# Weighting: reconstruction error, attention, simplex projection

def reconstruction_error(H, T, W, dim_normalize=False):
    r"""Per-node squared reconstruction error of one view.

    ``e_i = ||H_i - (T Wᵀ)_i||²``  (extension 2), optionally divided by
    the view dimensionality ``d`` (extension 3, dimension-normalised).

    Returns
    -------
    e : (n,) non-negative errors.
    """
    R = H - T @ W.T
    e = (R ** 2).sum(1)
    if dim_normalize:
        e = e / H.shape[1]
    return e


def view_attention(errors, lam=1.0, reg=None):
    r"""Node-level cross-view attention weights (extensions 2-5).

    Given per-view per-node errors ``errors`` (list of ``(n,)`` arrays,
    one per view), returns the fusion weights ``β`` (list of ``(n,)``
    arrays) with ``Σ_v β_{v,i} = 1`` for every node *i*.

    The weights solve, per node, the regularised allocation problem

        min_β  Σ_v β_v e_v + λ Ω(β)   s.t.  β ≥ 0, Σ_v β_v = 1

    whose closed form depends on the regulariser:

    * ``reg=None`` or ``"entropy"`` →  softmax:
      ``β_v = exp(-e_v/λ) / Σ_w exp(-e_w/λ)``.
      (Entropy regularisation is *exactly* temperature scaling — the
      ``λ`` plays the role of the softmax temperature.)
    * ``reg="l2"`` →  Euclidean projection of ``-e/(2λ)`` onto the
      probability simplex (can yield sparse / zero weights).

    Parameters
    ----------
    errors : list[np.ndarray]   per-view (n,) errors.
    lam : float                 regularisation strength λ (a.k.a. temperature).
    reg : {None, "entropy", "l2"}.
    """
    E = np.stack(errors, axis=1)           # (n, V)
    lam = max(float(lam), EPS)

    # Scale-invariant temperature: divide errors by their global median
    # so that ``lam`` is a *unitless relative* temperature.  Without this
    # the softmax sharpness depends on the absolute error magnitude (which
    # in turn depends on the feature scale and, after dimension
    # normalisation, on 1/d) — flattening the attention to near-uniform.
    scale = np.median(E[E > 0]) if np.any(E > 0) else 1.0
    E = E / max(scale, EPS)

    if reg == "l2":
        scores = -E / (2.0 * lam)          # (n, V)
        B = simplex_project(scores)        # row-wise simplex projection
    else:  # None or "entropy"  → softmax
        logits = -E / lam
        logits = logits - logits.max(axis=1, keepdims=True)
        ex = np.exp(logits)
        B = ex / np.maximum(ex.sum(axis=1, keepdims=True), EPS)

    return [B[:, v] for v in range(B.shape[1])]


def simplex_project(V):
    """Row-wise Euclidean projection onto the probability simplex.

    Implements the classic sort-based algorithm (Wang &
    Carreira-Perpiñán, 2013) for each row of ``V`` (shape ``(n, k)``).
    Used for the L2-regularised attention variant.
    """
    n, k = V.shape
    U = np.sort(V, axis=1)[:, ::-1]              # descending
    cssv = np.cumsum(U, axis=1) - 1.0
    ind = np.arange(1, k + 1)
    cond = U - cssv / ind > 0
    rho = cond.sum(axis=1)                        # (n,)
    theta = cssv[np.arange(n), rho - 1] / rho
    W = np.maximum(V - theta[:, None], 0.0)
    return W


# Robust ℓ2,1 loss (extension 1)

def irls_weights(H, T, W, normalize=True):
    r"""IRLS node weights for an ℓ2,1 reconstruction loss.

    Minimising ``Σ_i ||(H - T Wᵀ)_i||_2`` (ℓ2,1, robust to outlier
    nodes) instead of ``Σ_i ||·||²`` is equivalent to iteratively
    re-weighted least squares with per-node weight

        u_i = 1 / (2 ||(H - T Wᵀ)_i||_2 + eps).

    Nodes with large residuals (outliers / corrupted / misaligned)
    get *down-weighted*, which is the source of the robustness.

    Parameters
    ----------
    normalize : bool — rescale weights to mean 1 so the overall problem
        scale (and hence backward compatibility) is preserved.

    Returns
    -------
    u : (n,) positive weights.
    """
    R = H - T @ W.T
    row_norm = np.sqrt((R ** 2).sum(1))
    u = 1.0 / (2.0 * row_norm + EPS)
    if normalize:
        u = u / max(u.mean(), EPS)
    return u


# Controlled graph propagation (extension 6)

def controlled_propagation(H_norm, S, gamma):
    r"""Partial graph propagation ``H = ((1-γ)I + γ S) H_norm``.

    * ``γ = 0`` → raw (normalised) features, no propagation.
    * ``γ = 1`` → full propagation ``S H_norm`` (original LMGEC).
    * ``0 < γ < 1`` → convex blend (mitigates over-smoothing).
    """
    if S is None or gamma <= 0.0:
        return np.asarray(H_norm)
    SH = S @ H_norm
    if sp.issparse(SH):
        SH = SH.toarray()
    H_norm = np.asarray(H_norm)
    if gamma >= 1.0:
        return SH
    return (1.0 - gamma) * H_norm + gamma * SH


def graph_smoothness(H_norm, S):
    r"""Normalised graph-signal smoothness energy.

        Smooth = tr(Hᵀ L H) / ||H||_F²,   L = I − S.

    Low values → the signal already varies little over the graph
    (little to gain from propagation); high values → rough signal.
    Used by the data-driven γ heuristic in ``LMGECplus``.
    """
    H_norm = np.asarray(H_norm)
    SH = S @ H_norm
    if sp.issparse(SH):
        SH = SH.toarray()
    LH = H_norm - SH                       # L H = (I - S) H
    num = float((H_norm * LH).sum())       # tr(Hᵀ L H)
    den = float((H_norm ** 2).sum())
    return num / max(den, EPS)


def smoothness_to_gamma(smooth, lo=0.0, hi=1.0):
    """Map a smoothness energy in [0, ~1] to a γ in ``[lo, hi]``.

    Rougher signals (larger smoothness energy) get *more* propagation.
    The energy of an (I−S) operator lies in [0, ~2]; we clip/scale it
    into the requested γ range with a simple linear map.
    """
    g = float(np.clip(smooth, 0.0, 1.0))
    return lo + (hi - lo) * g


# Soft clustering (extension 7)

def soft_assign(Z, F, beta=1.0):
    r"""Softmax (soft K-means) responsibilities.

        G_{ij} = exp(-β d_ij) / Σ_l exp(-β d_il),   d = ||z_i - f_j||².

    ``β`` controls sharpness (large β → near-hard).  Returns an
    ``(n, k)`` row-stochastic responsibility matrix.
    """
    d2 = (
        (Z ** 2).sum(1)[:, None]
        - 2.0 * Z @ F.T
        + (F ** 2).sum(1)[None, :]
    )
    logits = -beta * d2
    logits = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / np.maximum(ex.sum(axis=1, keepdims=True), EPS)
