"""
MultiViewDataset — unified container for multi-view data.

Supports **tuple unpacking** for backward compatibility::

    Xs, As, y = load_acm()            # still works

as well as named attribute access::

    data = load_acm()
    data.Xs                            # list of feature matrices
    data.As                            # list of adjacency matrices (or Nones)
    data.labels                        # ground-truth label vector
    data.n_samples                     # number of samples
    data.n_views                       # number of views
    data.n_clusters                    # number of unique labels
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


class MultiViewDataset:
    """Container returned by every ``pymv`` dataset loader.

    Parameters
    ----------
    Xs : list
        Feature matrices, one per view.
    As : list
        Adjacency matrices, one per view.  Elements may be ``None``
        for feature-only datasets.
    labels : array-like
        Ground-truth cluster labels.
    name : str, optional
        Human-readable dataset name (e.g. ``"acm"``).

    Notes
    -----
    The class implements ``__iter__`` and ``__len__`` so that the
    classic ``Xs, As, y = load_<dataset>()`` idiom keeps working.
    """

    def __init__(
        self,
        Xs: list,
        As: list,
        labels,
        name: str | None = None,
    ):
        self.Xs = Xs
        self.As = As
        self.labels = np.asarray(labels).reshape(-1)
        self.name = name

    # ---- convenience properties ----------------------------------

    @property
    def n_samples(self) -> int:
        """Number of samples (rows of the first view)."""
        return int(self.Xs[0].shape[0])

    @property
    def n_views(self) -> int:
        """Number of views."""
        return len(self.Xs)

    @property
    def n_clusters(self) -> int:
        """Number of unique labels."""
        return int(len(np.unique(self.labels)))

    @property
    def feature_dims(self) -> list[int]:
        """Feature dimensionality of each view."""
        return [int(X.shape[1]) for X in self.Xs]

    @property
    def has_graphs(self) -> bool:
        """``True`` if at least one adjacency is not ``None``."""
        return any(A is not None for A in self.As)

    # ---- tuple unpacking (backward compat) -----------------------

    def __iter__(self) -> Iterator:
        """Yield ``(Xs, As, labels)`` for tuple unpacking."""
        yield self.Xs
        yield self.As
        yield self.labels

    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int):
        return (self.Xs, self.As, self.labels)[idx]

    # ---- display -------------------------------------------------

    def __repr__(self) -> str:
        name_str = f"'{self.name}'" if self.name else "unnamed"
        graph_str = "with graphs" if self.has_graphs else "no graphs"
        return (
            f"MultiViewDataset({name_str}, "
            f"n_samples={self.n_samples}, "
            f"n_views={self.n_views}, "
            f"n_clusters={self.n_clusters}, "
            f"dims={self.feature_dims}, "
            f"{graph_str})"
        )

    def summary(self) -> str:
        """Return a concise multi-line summary string."""
        lines = [
            f"Dataset     : {self.name or '(unnamed)'}",
            f"Samples     : {self.n_samples}",
            f"Views       : {self.n_views}",
            f"Clusters    : {self.n_clusters}",
            f"Feature dims: {self.feature_dims}",
            f"Has graphs  : {self.has_graphs}",
        ]
        return "\n".join(lines)
