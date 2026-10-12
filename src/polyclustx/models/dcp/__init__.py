"""Public DCP estimator exports."""

from .dcp import DCP

__all__ = ["DCP"]


def main():
    """Installed dcp command."""
    from polyclustx.command import model_main

    model_main("dcp")
