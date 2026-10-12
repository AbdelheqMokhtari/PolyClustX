"""Public CONAN estimator exports."""

from .conan import CONAN

__all__ = ["CONAN"]


def main():
    """Installed conan command."""
    from polyclustx.command import model_main

    model_main("conan")
