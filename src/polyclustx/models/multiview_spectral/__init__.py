"""Public MultiViewSpectralClustering estimator exports."""

from .multiview_spectral import MultiViewSpectralClustering, MultiviewSpectralClustering

__all__ = ["MultiViewSpectralClustering","MultiviewSpectralClustering"]


def main():
    """Installed multiview-spectral command."""
    from polyclustx.command import model_main

    model_main("multiview_spectral")
