"""
Clustering evaluation metrics for polyclustx.

This module provides utilities for evaluating clustering results
using both **external** (ground-truth-based) and **internal**
(intrinsic quality) metrics.

External metrics (require ground-truth labels)
----------------------------------------------
- :func:`clustering_accuracy` — Accuracy (ACC) via Hungarian
  alignment of the confusion matrix.
- :func:`clustering_f1_score` — F1 score (macro / micro / weighted)
  computed after Hungarian alignment.
- :func:`nmi_score` — Normalized Mutual Information (NMI).
- :func:`ari_score` — Adjusted Rand Index (ARI).

Internal metrics (no ground truth needed)
-----------------------------------------
- :func:`silhouette_score` — mean Silhouette Coefficient.
- :func:`calinski_harabasz_score` — Calinski–Harabasz Index
  (Variance Ratio Criterion).
- :func:`davies_bouldin_score` — Davies–Bouldin Index.

Internal helpers (used by the external metrics; usable but rarely needed)
-------------------------------------------------------------------------
- :func:`ordered_confusion_matrix` — confusion matrix reordered
  by the Hungarian algorithm to best match predicted ↔ true labels.
- :func:`cmat_to_pseudo_labels` — expand a confusion matrix back
  into aligned label vectors for sklearn metric functions.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

# Internal helpers


def ordered_confusion_matrix(y_true, y_pred):
    r"""Build a confusion matrix with columns reordered by Hungarian alignment.

    .. note:: **Internal helper** — called by
       :func:`clustering_accuracy` and :func:`clustering_f1_score`.
       You normally do not need to call this directly.

    Because cluster labels are arbitrary (cluster 0 in the prediction
    may correspond to class 2 in the ground truth), a raw confusion
    matrix is misleading.  This function solves the **optimal
    assignment** between predicted and true labels by minimising the
    total misclassification cost with the Hungarian algorithm
    (:func:`scipy.optimize.linear_sum_assignment`).

    Concretely, it builds the cost matrix

    .. math::
        C_{ij} = \max(M) - M_{ij}

    where :math:`M` is the standard confusion matrix, and then
    reorders both rows and columns according to the optimal
    assignment so that the diagonal captures the best possible
    label alignment.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted cluster labels.

    Returns
    -------
    conf_mat : np.ndarray of shape (n_classes, n_classes)
        Confusion matrix whose rows / columns are permuted to
        maximise the trace (i.e. the total number of correctly
        matched samples).

    See Also
    --------
    cmat_to_pseudo_labels : Convert the aligned matrix back to
        label vectors.
    clustering_accuracy : Uses this matrix to compute ACC.
    clustering_f1_score : Uses this matrix to compute F1.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 0, 1, 1, 2, 2])
    >>> y_pred = np.array([2, 2, 0, 0, 1, 1])   # permuted labels
    >>> cm = ordered_confusion_matrix(y_true, y_pred)
    >>> np.trace(cm) == len(y_true)               # perfect match
    True
    """
    conf_mat = metrics.confusion_matrix(y_true, y_pred)

    cost_matrix = np.max(conf_mat) - conf_mat
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    conf_mat = conf_mat[row_ind, :]
    conf_mat = conf_mat[:, col_ind]

    return conf_mat


def cmat_to_pseudo_labels(conf_mat):
    r"""Expand a confusion matrix into aligned label vectors.

    .. note:: **Internal helper** — called by
       :func:`clustering_f1_score`.  You normally do not need to
       call this directly.

    Given an :math:`K \times K` confusion matrix (typically already
    reordered by :func:`ordered_confusion_matrix`), this function
    reconstructs two label vectors ``y_true`` and ``y_pred`` such
    that ``sklearn.metrics.confusion_matrix(y_true, y_pred)`` would
    reproduce the input matrix.

    This is useful because many sklearn metric functions (e.g.
    ``f1_score``) expect flat label arrays rather than a
    pre-computed confusion matrix.

    Parameters
    ----------
    conf_mat : np.ndarray of shape (K, K)
        Confusion matrix, where ``conf_mat[i, j]`` is the count of
        samples with true class *i* assigned to predicted class *j*.

    Returns
    -------
    y_true : list[int]
        Reconstructed ground-truth labels.
    y_pred : list[int]
        Reconstructed predicted labels, aligned with *y_true*.

    See Also
    --------
    ordered_confusion_matrix : Produces the aligned confusion
        matrix consumed by this function.

    Examples
    --------
    >>> import numpy as np
    >>> cm = np.array([[3, 1],
    ...                [0, 4]])
    >>> y_true, y_pred = cmat_to_pseudo_labels(cm)
    >>> len(y_true) == cm.sum()
    True
    """
    y_true = []
    y_pred = []

    for true_class, row in enumerate(conf_mat):
        for pred_class, count in enumerate(row):
            y_true.extend([true_class] * count)
            y_pred.extend([pred_class] * count)

    return y_true, y_pred

# External metrics (require ground truth)


def clustering_accuracy(y_true, y_pred):
    r"""Clustering accuracy (ACC) with Hungarian label alignment.

    **Public API** — the standard accuracy metric for unsupervised
    clustering evaluation.

    Cluster labels are arbitrary, so a direct comparison with the
    ground truth would be meaningless.  This function first finds
    the **optimal one-to-one mapping** between predicted clusters
    and true classes via the Hungarian algorithm, then computes

    .. math::
        \text{ACC} = \frac{\operatorname{tr}(\hat{M})}
                          {\sum_{ij} \hat{M}_{ij}}

    where :math:`\hat{M}` is the confusion matrix after optimal
    row / column permutation (see
    :func:`ordered_confusion_matrix`).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted cluster labels.

    Returns
    -------
    acc : float
        Accuracy in ``[0, 1]``.  Higher is better.

    See Also
    --------
    clustering_f1_score : F1 score with the same alignment step.
    nmi_score : Normalized Mutual Information (alignment-free).
    ari_score : Adjusted Rand Index (alignment-free).

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 0, 1, 1, 2, 2])
    >>> y_pred = np.array([1, 1, 2, 2, 0, 0])  # permuted but correct
    >>> clustering_accuracy(y_true, y_pred)
    1.0
    """
    conf_mat = ordered_confusion_matrix(y_true, y_pred)

    return np.trace(conf_mat) / np.sum(conf_mat)


def clustering_f1_score(y_true, y_pred, average="macro"):
    r"""F1 score for clustering with Hungarian label alignment.

    **Public API** — computes the F1 score after optimally
    matching predicted cluster IDs to ground-truth classes.

    The alignment step is identical to :func:`clustering_accuracy`:
    the confusion matrix is reordered with the Hungarian algorithm
    (see :func:`ordered_confusion_matrix`), then pseudo-labels are
    reconstructed (see :func:`cmat_to_pseudo_labels`) and passed
    to :func:`sklearn.metrics.f1_score`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted cluster labels.
    average : {'macro', 'micro', 'weighted', 'binary'}, default='macro'
        Averaging strategy forwarded to
        :func:`sklearn.metrics.f1_score`.

        - ``'macro'`` — unweighted mean of per-class F1.
        - ``'micro'`` — global precision / recall (equivalent to
          accuracy after alignment).
        - ``'weighted'`` — class-size-weighted mean.

    Returns
    -------
    f1 : float
        F1 score in ``[0, 1]``.  Higher is better.

    See Also
    --------
    clustering_accuracy : Simpler accuracy metric with the same
        alignment procedure.
    nmi_score : Alignment-free information-theoretic metric.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 0, 1, 1, 2, 2])
    >>> y_pred = np.array([1, 1, 2, 2, 0, 0])
    >>> f1 = clustering_f1_score(y_true, y_pred)
    >>> f1  # perfect clustering → 1.0
    1.0
    """
    conf_mat = ordered_confusion_matrix(y_true, y_pred)

    pseudo_y_true, pseudo_y_pred = cmat_to_pseudo_labels(conf_mat)

    return metrics.f1_score(
        pseudo_y_true,
        pseudo_y_pred,
        average=average
    )


def nmi_score(y_true, y_pred):
    r"""Normalized Mutual Information (NMI) between two clusterings.

    **Public API** — thin wrapper around
    :func:`sklearn.metrics.normalized_mutual_info_score`.

    NMI measures the amount of information shared between the
    predicted clustering and the ground truth, normalised to
    ``[0, 1]``:

    .. math::
        \text{NMI}(U, V)
          = \frac{2\,I(U;\,V)}{H(U) + H(V)}

    where :math:`I` is mutual information and :math:`H` is entropy.
    Unlike ACC or F1, NMI does **not** require label alignment
    because it operates on set partitions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted cluster labels.

    Returns
    -------
    nmi : float
        Score in ``[0, 1]``.  ``1`` indicates perfect agreement,
        ``0`` indicates independence.

    See Also
    --------
    ari_score : Another alignment-free metric based on pair
        counting.
    clustering_accuracy : Accuracy via Hungarian alignment.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([1, 1, 0, 0])
    >>> nmi_score(y_true, y_pred)
    1.0
    """
    return metrics.normalized_mutual_info_score(
        y_true,
        y_pred
    )


def ari_score(y_true, y_pred):
    r"""Adjusted Rand Index (ARI) between two clusterings.

    **Public API** — thin wrapper around
    :func:`sklearn.metrics.adjusted_rand_score`.

    The Rand Index counts the fraction of sample pairs that are
    either in the same cluster in both partitions or in different
    clusters in both.  ARI corrects for chance:

    .. math::
        \text{ARI}
          = \frac{\text{RI} - \mathbb{E}[\text{RI}]}
                 {\max(\text{RI}) - \mathbb{E}[\text{RI}]}

    A score of ``1`` means perfect agreement, ``0`` means the
    agreement is no better than random, and negative values
    indicate worse-than-random.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted cluster labels.

    Returns
    -------
    ari : float
        Score in ``[-1, 1]``.  ``1`` = perfect, ``0`` = random.

    See Also
    --------
    nmi_score : Information-theoretic alignment-free metric.
    clustering_accuracy : Accuracy via Hungarian alignment.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([1, 1, 0, 0])
    >>> ari_score(y_true, y_pred)
    1.0
    """
    return metrics.adjusted_rand_score(
        y_true,
        y_pred
    )


# Internal metrics (no ground truth needed)

def silhouette_score(X, labels):
    r"""Mean Silhouette Coefficient over all samples.

    **Public API** — thin wrapper around
    :func:`sklearn.metrics.silhouette_score`.

    The Silhouette Coefficient for a single sample is

    .. math::
        s = \frac{b - a}{\max(a,\, b)}

    where *a* is the mean intra-cluster distance and *b* is the
    mean nearest-cluster distance.  The overall score is the mean
    of *s* across all samples.

    This is an **internal** metric: it evaluates cluster quality
    from the data geometry alone, without ground-truth labels.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix (or precomputed distance matrix).
    labels : array-like of shape (n_samples,)
        Cluster assignments.

    Returns
    -------
    score : float
        Mean Silhouette Coefficient in ``[-1, 1]``.  Higher is
        better; values near ``+1`` indicate well-separated
        clusters.

    See Also
    --------
    calinski_harabasz_score : Variance-ratio internal metric.
    davies_bouldin_score : Average cluster-similarity internal
        metric (lower is better).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0, 0], [1, 0], [10, 10], [11, 10]])
    >>> labels = np.array([0, 0, 1, 1])
    >>> score = silhouette_score(X, labels)
    >>> score > 0.5
    True
    """
    return metrics.silhouette_score(
        X,
        labels
    )


def davies_bouldin_score(X, labels):
    r"""Davies–Bouldin Index.

    **Public API** — thin wrapper around
    :func:`sklearn.metrics.davies_bouldin_score`.

    The Davies–Bouldin Index is defined as the average, over all
    clusters, of the maximum similarity between each cluster and
    any other cluster:

    .. math::
        \text{DB} = \frac{1}{K}\sum_{i=1}^{K}
            \max_{j \neq i}
            \frac{S_i + S_j}{d(c_i,\, c_j)}

    where :math:`S_i` is the mean distance from points in cluster
    *i* to their centroid :math:`c_i`.

    This is an **internal** metric (no ground truth needed).
    **Lower values indicate better clustering.**

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    labels : array-like of shape (n_samples,)
        Cluster assignments.

    Returns
    -------
    score : float
        Davies–Bouldin Index ≥ 0.  Lower is better; ``0`` means
        perfectly separated clusters.

    See Also
    --------
    silhouette_score : Silhouette-based internal metric (higher is
        better).
    calinski_harabasz_score : Variance-ratio internal metric
        (higher is better).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0, 0], [1, 0], [10, 10], [11, 10]])
    >>> labels = np.array([0, 0, 1, 1])
    >>> score = davies_bouldin_score(X, labels)
    >>> score < 1.0
    True
    """
    return metrics.davies_bouldin_score(
        X,
        labels
    )


def calinski_harabasz_score(X, labels):
    r"""Calinski–Harabasz Index (Variance Ratio Criterion).

    **Public API** — thin wrapper around
    :func:`sklearn.metrics.calinski_harabasz_score`.

    Also known as the Variance Ratio Criterion, it is the ratio of
    between-cluster dispersion to within-cluster dispersion:

    .. math::
        \text{CH} = \frac{\operatorname{tr}(B_K)}
                         {\operatorname{tr}(W_K)}
                    \times \frac{N - K}{K - 1}

    where :math:`B_K` and :math:`W_K` are the between- and
    within-cluster scatter matrices, :math:`N` is the number of
    samples, and :math:`K` is the number of clusters.

    This is an **internal** metric (no ground truth needed).
    **Higher values indicate better-defined clusters.**

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    labels : array-like of shape (n_samples,)
        Cluster assignments.

    Returns
    -------
    score : float
        Calinski–Harabasz Index ≥ 0.  Higher is better.

    See Also
    --------
    silhouette_score : Silhouette-based internal metric (higher is
        better).
    davies_bouldin_score : Average cluster-similarity internal
        metric (lower is better).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0, 0], [1, 0], [10, 10], [11, 10]])
    >>> labels = np.array([0, 0, 1, 1])
    >>> score = calinski_harabasz_score(X, labels)
    >>> score > 1.0
    True
    """
    return metrics.calinski_harabasz_score(
        X,
        labels
    )
