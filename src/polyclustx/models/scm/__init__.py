"""Simple Contrastive Multi-view Clustering (SCM)."""

from .scm import SCM

__all__ = ["SCM"]


def main():
    """Installed scm command."""
    from polyclustx.command import model_main

    model_main("scm")
