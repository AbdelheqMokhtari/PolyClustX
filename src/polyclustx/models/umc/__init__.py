"""Public UMC estimator exports."""

from .umc import UMC

__all__ = ["UMC"]


def main():
    """Installed umc command."""
    from polyclustx.command import model_main

    model_main("umc")
