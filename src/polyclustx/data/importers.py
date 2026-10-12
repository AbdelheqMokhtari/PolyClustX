"""Explicit, shared imports for Python and the interactive workbench."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import loadmat

from .dataset import MultiViewDataset
from .missing import align_views


def read_arrays(source, *, filename=None, delimiter=",", skip_header="auto"):
    """Inspect NPY, NPZ, numeric CSV/TSV or MATLAB arrays without guessing roles.

    ``source`` is a path or bytes with an explicit filename. NumPy pickles are
    disabled. MATLAB cells are exposed as ``key[0]``, ``key[1]``, etc., in
    column-major order. MATLAB v7.3/HDF5 is not supported; export v7 or NPZ.
    CSV/TSV files automatically skip one header row when every nonempty field
    in the first row is not numeric. Pass ``skip_header=0`` or ``1`` to
    override detection. String ID columns should be stored in separate
    NPY/NPZ vectors, not numeric feature CSVs.
    """
    name = filename or str(source)
    suffix = Path(name).suffix.lower()
    stream = io.BytesIO(source) if isinstance(source, bytes) else source
    if suffix == ".npy":
        return {Path(name).stem: np.load(stream, allow_pickle=False)}
    if suffix == ".npz":
        with np.load(stream, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    if suffix in {".csv", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else delimiter
        if skip_header == "auto":
            if isinstance(source, bytes):
                first_line = source.decode("utf-8-sig", errors="replace").splitlines()[0]
            else:
                with Path(source).open(encoding="utf-8-sig", errors="replace") as handle:
                    first_line = handle.readline()
            fields = [field.strip() for field in first_line.split(separator) if field.strip()]
            try:
                for field in fields:
                    float(field)
                skip_header = 0
            except ValueError:
                skip_header = 1
        if skip_header not in {0, 1}:
            raise ValueError("CSV header override must be 0, 1, or 'auto'")
        return {Path(name).stem: np.genfromtxt(stream, delimiter=separator, skip_header=skip_header, ndmin=2)}
    if suffix == ".mat":
        try:
            raw = loadmat(stream)
        except NotImplementedError as exc:
            raise ValueError("MATLAB v7.3 is unsupported; export MATLAB v7 or NPZ") from exc
        out = {}

        def expose(key, value):
            if isinstance(value, np.ndarray) and value.dtype == object:
                for i, cell in enumerate(value.ravel(order="F")):
                    expose(f"{key}[{i}]", cell)
            else:
                out[key] = value

        for key, value in raw.items():
            if not key.startswith("__"):
                expose(key, value)
        return out
    raise ValueError("Supported files: .npy, .npz, .csv, .tsv, .mat (v7 or earlier)")


def dataset_from_arrays(arrays, *, view_keys, graph_keys=None, label_key=None, label_column=None,
                        id_keys=None, label_id_key=None, mask_key=None,
                        transpose_views=(), detect_nan_rows=False, modalities=None, name="uploaded"):
    """Build a dataset using explicit array roles, shared by all file formats.

    graph_keys maps selected view keys to lists of relation keys. With id_keys,
    graphs follow each view's local sample order. Without IDs the user asserts
    that every selected matrix has the same row order. Fully NaN rows may be
    detected only when explicitly enabled; partial NaNs are rejected. An
    explicit mask uses global row order and cannot be combined with ID alignment.

    ``label_column=(view_key, column_index)`` extracts labels from one feature
    array and removes that column before fitting. The index is zero-based,
    after any requested transpose. It cannot be combined with ``label_key``.
    """
    if not view_keys or len(set(view_keys)) != len(view_keys):
        raise ValueError("Select at least one unique view key")
    if label_key in view_keys:
        raise ValueError("A labels array cannot also be selected as a feature view")
    graph_keys = graph_keys or {}
    if set(graph_keys) - set(view_keys):
        raise ValueError("Every graph must be mapped to a selected view")
    Xs = [arrays[k].T if k in transpose_views else arrays[k] for k in view_keys]
    labels = None if label_key is None else arrays[label_key]
    if label_column is not None:
        if label_key is not None:
            raise ValueError("Choose a labels array or an embedded label column, not both")
        if not isinstance(label_column, (list, tuple)) or len(label_column) != 2:
            raise ValueError("label_column must contain a view key and a zero-based column index")
        key, column = label_column
        if key not in view_keys:
            raise ValueError("The label column must belong to a selected feature view")
        view = view_keys.index(key)
        matrix = Xs[view]
        if (len(matrix.shape) != 2 or isinstance(column, bool)
                or not isinstance(column, (int, np.integer)) or not 0 <= column < matrix.shape[1]):
            raise ValueError("Select a valid zero-based label column")
        if matrix.shape[1] < 2:
            raise ValueError("The label column must leave at least one feature column")
        if sparse.issparse(matrix):
            matrix = matrix.tocsr()
            labels = matrix[:, column].toarray().ravel()
            Xs[view] = matrix[:, np.arange(matrix.shape[1]) != column]
        else:
            labels = np.asarray(matrix)[:, column].copy()
            Xs[view] = np.delete(matrix, column, axis=1)
    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim not in {1, 2} or (labels.ndim == 2 and 1 not in labels.shape):
            raise ValueError("Labels must be a vector with one value per sample")
    Xs = [x if sparse.issparse(x) else np.asarray(x, dtype=float) for x in Xs]
    As = [[arrays[k] for k in graph_keys.get(view, [])] or None for view in view_keys]
    options = {"name": name, "view_names": view_keys, "modalities": modalities,
               "metadata": {"import": {"view_keys": list(view_keys), "graph_keys": graph_keys,
                                       "label_key": label_key, "label_column": label_column, "id_keys": id_keys,
                                       "transpose_views": list(transpose_views)}}}
    if id_keys:
        if len(id_keys) != len(view_keys) or mask_key is not None or detect_nan_rows:
            raise ValueError("ID alignment needs one ID key per view; do not combine with row masks/NaN detection")
        return align_views(Xs, [arrays[k] for k in id_keys], graphs=As, labels=labels,
                           label_ids=None if label_id_key is None else arrays[label_id_key], **options)
    mask = None if mask_key is None else np.asarray(arrays[mask_key])
    if mask is not None:
        if not np.isin(mask, [0, 1]).all():
            raise ValueError("Observation masks must contain only zero/one or boolean values")
        mask = mask.astype(bool)
    if detect_nan_rows:
        inferred = np.column_stack([np.ones(x.shape[0], dtype=bool) if sparse.issparse(x)
                                    else ~np.isnan(x).all(axis=1) for x in Xs])
        mask = inferred if mask is None else mask & inferred
    return MultiViewDataset(Xs, As, labels, observed_mask=mask, **options)


def load_files(paths, **mapping):
    """Read multiple files and map explicit namespaced keys into one dataset.

    Keys are ``filename:key``; for example ``view.csv:view`` or ``data.npz:X``.
    Duplicate basenames/keys are rejected rather than overwritten.
    """
    arrays = {}
    for path in paths:
        for key, value in read_arrays(path).items():
            full_key = f"{Path(path).name}:{key}"
            if full_key in arrays:
                raise ValueError(f"Duplicate array key: {full_key}")
            arrays[full_key] = value
    return dataset_from_arrays(arrays, **mapping)
