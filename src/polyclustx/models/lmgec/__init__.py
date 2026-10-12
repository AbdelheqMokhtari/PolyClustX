from .lmgec import LMGEC

__all__ = ["LMGEC"]


def main():
    """Installed lmgec command."""
    from polyclustx.command import model_main

    model_main("lmgec")
