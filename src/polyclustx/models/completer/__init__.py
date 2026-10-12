"""Public COMPLETER estimator exports."""

from .completer import COMPLETER, Completer

__all__ = ["COMPLETER","Completer"]


def main():
    """Installed completer command."""
    from polyclustx.command import model_main

    model_main("completer")
