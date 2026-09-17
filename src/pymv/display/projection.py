"""Reproducible visual projections, independent of the interface runtime."""

import numpy as np


def project_embedding(embedding, method="PCA", *, random_state=0, max_samples=5000):
    """Return a 2D projection and the selected row indices.

    Large embeddings are sampled uniformly for visualization only. Scores and
    clustering outputs are unchanged. UMAP is an optional dependency.
    """
    z = np.asarray(embedding)
    if z.ndim != 2 or z.shape[0] < 3 or not np.isfinite(z).all():
        raise ValueError("Projection requires a finite embedding with at least three samples")
    rows = np.arange(len(z))
    if len(z) > max_samples:
        rows = np.sort(np.random.default_rng(random_state).choice(len(z), max_samples, replace=False))
    z = z[rows]
    if method == "PCA":
        from sklearn.decomposition import PCA

        if z.shape[1] < 2:
            raise ValueError("PCA requires at least two embedding dimensions")
        projected = PCA(n_components=2, random_state=random_state).fit_transform(z)
    elif method == "t-SNE":
        from sklearn.manifold import TSNE

        projected = TSNE(
            n_components=2, perplexity=min(30, len(z) - 1), init="random", random_state=random_state
        ).fit_transform(z)
    elif method == "UMAP":
        try:
            from umap import UMAP
        except ImportError as exc:
            raise ValueError('Install UMAP support: pip install "pymv-clustering[umap]"') from exc
        projected = UMAP(
            n_components=2, n_neighbors=min(15, len(z) - 1), init="random", random_state=random_state
        ).fit_transform(z)
    else:
        raise ValueError("Choose PCA, t-SNE or UMAP")
    return projected, rows
