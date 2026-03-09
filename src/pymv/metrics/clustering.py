"""
Clustering evaluation metrics used in Pymv.

This module provides utilities for evaluating clustering results
with both external (ground-truth based) and internal metrics.

External metrics:
    - Accuracy (ACC) using Hungarian alignment
    - F1 score
    - Normalized Mutual Information (NMI)
    - Adjusted Rand Index (ARI)

Internal metrics:
    - Silhouette score
    - Calinski–Harabasz score
    - Davies–Bouldin score
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

# ------------------------------------------------
# Internal helpers
# ------------------------------------------------

def ordered_confusion_matrix(y_true, y_pred):

    conf_mat = metrics.confusion_matrix(y_true, y_pred)

    cost_matrix = np.max(conf_mat) - conf_mat
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    conf_mat = conf_mat[row_ind, :]
    conf_mat = conf_mat[:, col_ind]

    return conf_mat



def cmat_to_pseudo_labels(conf_mat):

    y_true = []
    y_pred = []

    for true_class, row in enumerate(conf_mat):
        for pred_class, count in enumerate(row):
            y_true.extend([true_class] * count)
            y_pred.extend([pred_class] * count)

    return y_true, y_pred

# ------------------------------------------------
# External metrics (require ground truth)
# ------------------------------------------------

def clustering_accuracy(y_true, y_pred):

    conf_mat = ordered_confusion_matrix(y_true, y_pred)

    return np.trace(conf_mat) / np.sum(conf_mat)




def clustering_f1_score(y_true, y_pred, average="macro"):

    conf_mat = ordered_confusion_matrix(y_true, y_pred)

    pseudo_y_true, pseudo_y_pred = cmat_to_pseudo_labels(conf_mat)

    return metrics.f1_score(
        pseudo_y_true,
        pseudo_y_pred,
        average=average
    )


def nmi_score(y_true, y_pred):

    return metrics.normalized_mutual_info_score(
        y_true,
        y_pred
    )


def ari_score(y_true, y_pred):

    return metrics.adjusted_rand_score(
        y_true,
        y_pred
    )


# ------------------------------------------------
# Internal metrics (no ground truth needed)
# ------------------------------------------------

def silhouette_score(X, labels):

    return metrics.silhouette_score(
        X,
        labels
    )


def davies_bouldin_score(X, labels):

    return metrics.davies_bouldin_score(
        X,
        labels
    )


def calinski_harabasz_score(X, labels):

    return metrics.calinski_harabasz_score(
        X,
        labels
    )