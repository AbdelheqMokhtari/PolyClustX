"""
Thin entry-point wrapper for pymv-cli.

This module is loaded by the console-script entry point **before**
the ``pymv`` package is imported.  That gives us a chance to set
``TF_CPP_MIN_LOG_LEVEL`` and other environment variables so that
TensorFlow's C++ runtime respects them when its shared library is
first loaded.
"""
from __future__ import annotations

import os
import sys


def _prescan_verbose() -> int:
    """Extract --verbose / -v from sys.argv without full arg-parsing."""
    raw = sys.argv[1:]
    for i, tok in enumerate(raw):
        if tok in ("-v", "--verbose") and i + 1 < len(raw):
            try:
                return int(raw[i + 1])
            except ValueError:
                pass
        if tok.startswith("-v") and len(tok) == 3 and tok[2:].isdigit():
            return int(tok[2:])
    return 1  # default


def main() -> None:
    level = _prescan_verbose()

    # Set C++ env vars *before* TF .so is loaded
    if level <= 2:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        os.environ["GRPC_VERBOSITY"] = "ERROR"
        os.environ["ABSL_MIN_LOG_LEVEL"] = "4"     # suppress abseil C++ logs

    if level <= 2:
        # Redirect the actual OS-level stderr file descriptor to
        # /dev/null.  TF's C++ code writes directly to fd 2, so
        # Python's sys.stderr redirect is not enough.
        #
        # level 0-2: suppress C++ noise for the entire lifetime
        # level 3  : let everything through
        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        _saved_fd = os.dup(2)
        os.dup2(_devnull_fd, 2)
        os.close(_devnull_fd)
        try:
            from pymv.cli import main as _real_main
            _real_main()
        finally:
            os.dup2(_saved_fd, 2)
            os.close(_saved_fd)
    else:
        from pymv.cli import main as _real_main
        _real_main()


if __name__ == "__main__":
    main()
