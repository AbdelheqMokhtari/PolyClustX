"""Public DGF estimator exports."""

from .dgf import DGF

__all__ = ["DGF"]


def main():
    """Installed dgf command."""
    from polyclustx.command import model_main

    model_main("dgf")
