"""Disk-backed run arrays: session history need not retain every embedding in RAM."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np


class StoredArtifact(Mapping):
    """Read one named array on demand from a trusted session-generated NPZ."""

    def __init__(self, path, arrays):
        self.path = Path(path)
        safe = {key: np.asarray(value) for key, value in arrays.items()}
        # String/object labels must not require enabling NumPy pickle loading.
        safe = {key: value.astype(str) if value.dtype.hasobject else value for key, value in safe.items()}
        np.savez_compressed(self.path, **safe)
        self.keys_ = tuple(safe)

    def __getitem__(self, key):
        if key not in self.keys_:
            raise KeyError(key)
        with np.load(self.path, allow_pickle=False) as archive:
            return archive[key]

    def __iter__(self):
        return iter(self.keys_)

    def __len__(self):
        return len(self.keys_)

    def __contains__(self, key):
        return key in self.keys_
