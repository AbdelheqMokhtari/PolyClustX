"""Public GCoMVKM estimator exports."""

from .g_comvkm import GCoMVKM

__all__ = ["GCoMVKM"]


def main():
    """Installed g-comvkm command."""
    from polyclustx.command import model_main

    model_main("g_comvkm")
