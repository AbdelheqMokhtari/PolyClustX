"""Abstract base class shared by every pymv model."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Common interface for multi-view clustering models.

    Subclasses implement :meth:`fit`; :meth:`fit_predict` is derived
    from it and only needs overriding when a model can predict more
    cheaply than it fits.

    After a successful ``fit`` a model is expected to expose:

    ``labels_``
        ``np.ndarray`` of shape ``(n_samples,)`` — the cluster
        assignment of each sample.
    ``embedding_``
        ``np.ndarray`` of shape ``(n_samples, embedding_dim)`` — the
        learned consensus embedding.
    """

    @abstractmethod
    def fit(self, Xs, As=None):
        """Fit the model on a list of views.

        Parameters
        ----------
        Xs : list[np.ndarray]
            Feature matrices — one per view.
        As : list or None, default=None
            Adjacency matrices — one per view, when the model uses
            graph structure.

        Returns
        -------
        self
        """
        raise NotImplementedError

    def fit_predict(self, Xs, As=None):
        """Fit the model and return the cluster labels.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        return self.fit(Xs, As).labels_
