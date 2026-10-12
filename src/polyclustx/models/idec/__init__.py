"""Public IDEC estimator exports."""

from .idec import IDEC

__all__ = ["IDEC"]


def main():
    """Installed idec command."""
    from polyclustx.command import model_main

    model_main("idec")
