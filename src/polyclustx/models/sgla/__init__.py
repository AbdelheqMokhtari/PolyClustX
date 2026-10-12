"""Public SGLA estimator exports."""

from .sgla import SGLA

__all__ = ["SGLA"]


def main():
    """Installed sgla command."""
    from polyclustx.command import model_main

    model_main("sgla")
