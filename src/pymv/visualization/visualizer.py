import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_loss(
    loss_history,
    figsize=(8, 5),
    title="Training Loss",
    xlabel="Iteration",
    ylabel="Loss",
    color="#2563eb",
    linewidth=2.0,
    marker=None,
    grid=True,
    log_scale=False,
    ax=None,
    show=True,
):
    """
    Plot the training loss curve.

    Parameters
    ----------
    loss_history : array-like
        1-D array of loss values (one per iteration).
        Typically ``model.loss_history_`` after calling
        :meth:`LMGEC.fit`.

    figsize : tuple of int, default ``(8, 5)``
        Figure size (width, height) in inches.
        Ignored when *ax* is provided.

    title : str, default ``"Training Loss"``
        Plot title.

    xlabel : str, default ``"Iteration"``
        X-axis label.

    ylabel : str, default ``"Loss"``
        Y-axis label.

    color : str, default ``"#2563eb"``
        Line colour (any Matplotlib colour spec).

    linewidth : float, default ``2.0``
        Line width.

    marker : str or None, default ``None``
        Marker style for each data point (e.g. ``"o"``).

    grid : bool, default ``True``
        Show background grid.

    log_scale : bool, default ``False``
        If ``True``, use a logarithmic y-axis.

    ax : matplotlib.axes.Axes or None
        Axes to draw on.  If ``None``, a new figure is created.

    show : bool, default ``True``
        Call ``plt.show()`` at the end.  Set to ``False`` when
        composing multi-panel figures.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object (useful for further customisation).

    Examples
    --------
    >>> from pymv.models import LMGEC
    >>> model = LMGEC(n_clusters=3, embedding_dim=4, max_iter=20)
    >>> model.fit(Xs, As)
    >>> from pymv.visualization import plot_loss
    >>> plot_loss(model.loss_history_)

    Side-by-side loss + clusters::

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_loss(model.loss_history_, ax=axes[0], show=False)
        plot_clusters(model.embedding_, labels=model.labels_, ax=axes[1])
        plt.tight_layout()
        plt.show()
    """
    loss = np.asarray(loss_history).ravel()
    iterations = np.arange(1, len(loss) + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        iterations, loss,
        color=color,
        linewidth=linewidth,
        marker=marker,
        markersize=4 if marker else None,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if log_scale:
        ax.set_yscale("log")

    if grid:
        ax.grid(True, alpha=0.3)

    ax.set_xlim(1, len(loss))

    if show:
        plt.show()

    return ax


def plot_clusters(
    X,
    labels=None,
    method="tsne",
    n_components=2,
    random_state=42,
    figsize=(8, 6),
    cmap="tab10",
    title=None,
    ax=None,
    show=True,
):
    """
    Visualize clusters using dimensionality reduction.

    Parameters
    ----------
    X : ndarray
        High dimensional embedding.

    labels : ndarray
        Cluster labels.

    method : {"tsne", "pca", "umap"}
        Reduction method.

    n_components : int
        Target dimension (default 2).

    random_state : int
        Random seed.

    figsize : tuple
        Figure size.  Ignored when *ax* is provided.

    cmap : str
        Matplotlib colormap.

    title : str
        Plot title.

    ax : matplotlib.axes.Axes or None
        Axes to draw on.  If ``None``, a new figure is created.

    show : bool, default ``True``
        Call ``plt.show()`` at the end.  Set to ``False`` when
        composing multi-panel figures.

    Returns
    -------
    ndarray
        The reduced coordinates (n_samples, n_components).
    """

    method = method.lower()

    if method == "tsne":

        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )

    elif method == "pca":

        reducer = PCA(
            n_components=n_components,
            random_state=random_state
        )

    elif method == "umap":

        try:
            import umap
        except ImportError:
            raise ImportError(
                "UMAP is not installed. Run: pip install umap-learn"
            )

        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state
        )

    else:
        raise ValueError(
            "method must be one of: 'tsne', 'pca', 'umap'"
        )

    X_reduced = reducer.fit_transform(X)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:

        labels = np.asarray(labels)
        unique_labels = np.unique(labels)

        colormap = plt.get_cmap(cmap)
        colors = colormap(np.linspace(0, 1, len(unique_labels)))

        for color, lab in zip(colors, unique_labels):

            idx = labels == lab

            ax.scatter(
                X_reduced[idx, 0],
                X_reduced[idx, 1],
                s=20,
                color=color,
                label=f"Cluster {lab}",
                alpha=0.8,
            )

        ax.legend()

    else:

        ax.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            s=20,
            alpha=0.8,
        )

    if title is None:
        title = f"{method.upper()} cluster visualization"

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    if show:
        plt.show()

    return X_reduced
