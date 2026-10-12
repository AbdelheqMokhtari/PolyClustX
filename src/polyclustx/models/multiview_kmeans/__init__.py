"""Public MultiViewKMeans estimator exports."""

from .multiview_kmeans import MultiViewKMeans, MultiviewKMeans

__all__ = ["MultiViewKMeans","MultiviewKMeans"]


def main():
    """Installed multiview-kmeans command."""
    from polyclustx.command import model_main

    model_main("multiview_kmeans")
