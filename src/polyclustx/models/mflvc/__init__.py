"""Public MFLVC estimator exports."""

from .mflvc import MFLVC

__all__ = ["MFLVC"]


def main():
    """Installed mflvc command."""
    from polyclustx.command import model_main

    model_main("mflvc")
