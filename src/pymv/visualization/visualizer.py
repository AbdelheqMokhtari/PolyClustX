import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_clusters(
    X,
    labels=None,
    method="tsne",
    n_components=2,
    random_state=42,
    figsize=(8, 6),
    cmap="tab10",
    title=None,
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
        Figure size.

    cmap : str
        Matplotlib colormap.

    title : str
        Plot title.
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

    plt.figure(figsize=figsize)

    if labels is not None:

        labels = np.asarray(labels)
        unique_labels = np.unique(labels)

        colormap = plt.get_cmap(cmap)
        colors = colormap(np.linspace(0, 1, len(unique_labels)))

        for color, lab in zip(colors, unique_labels):

            idx = labels == lab

            plt.scatter(
                X_reduced[idx, 0],
                X_reduced[idx, 1],
                s=20,
                color=color,
                label=f"Cluster {lab}",
                alpha=0.8,
            )

        plt.legend()

    else:

        plt.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            s=20,
            alpha=0.8,
        )

    if title is None:
        title = f"{method.upper()} cluster visualization"

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.show()

    return X_reduced