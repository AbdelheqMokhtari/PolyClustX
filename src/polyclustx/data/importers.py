"""Explicit, shared imports for Python and the interactive workbench."""

from __future__ import annotations

import io
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import loadmat

from .dataset import MultiViewDataset
from .missing import align_views


def array_roles(arrays):
    """Find standardized X* features, A* relations and the exact labels key.

    Namespaced keys from ``load_files`` are supported. Natural ordering puts
    X2 before X10. Other arrays are never interpreted as training features.
    """
    def local(key):
        return key.rsplit(":", 1)[-1]

    def order(key):
        return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", key)]

    return {
        "features": sorted((k for k in arrays if local(k).startswith("X")), key=order),
        "graphs": sorted((k for k in arrays if local(k).startswith("A")), key=order),
        "labels": [k for k in arrays if local(k) == "labels"],
    }


def default_views(arrays):
    """Build explicit defaults for the standardized dataset layout.

    A shared feature with several graphs gives one view per graph. Otherwise
    each feature gives one view and receives all supplied relations. Pass a
    named ``views`` mapping to choose different combinations.
    """
    roles = array_roles(arrays)
    features, graphs = roles["features"], roles["graphs"]
    if not features:
        raise ValueError("No X* feature arrays found. Select view_keys explicitly for a nonstandard file.")
    if len(features) == 1 and len(graphs) > 1:
        return {f"{features[0]} + {graph}": (features[0], [graph]) for graph in graphs}
    return {feature: (feature, list(graphs)) for feature in features}


def _selection(arrays, views):
    if not isinstance(views, Mapping) or not views:
        raise ValueError("views must be a nonempty mapping of view names to (feature, graphs) pairs")
    features, relations, names = [], [], []
    for name, value in views.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each view needs a nonempty name")
        if isinstance(value, str):
            feature, graphs = value, []
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            feature, graphs = value
            graphs = [] if graphs is None else [graphs] if isinstance(graphs, str) else graphs
        else:
            raise ValueError("Each view must be a feature key or a (feature key, graph keys) pair")
        if not isinstance(feature, str) or feature not in arrays:
            raise ValueError(f"Unknown feature array: {feature!r}")
        if not isinstance(graphs, (list, tuple)) or any(not isinstance(k, str) or k not in arrays for k in graphs):
            raise ValueError(f"Unknown or invalid graph arrays for view {name!r}")
        if len(set(graphs)) != len(graphs):
            raise ValueError("A graph relation cannot appear twice in the same view")
        if feature.rsplit(":", 1)[-1] == "labels" or any(k.rsplit(":", 1)[-1] == "labels" for k in graphs):
            raise ValueError("The labels array is for evaluation, not features or graphs")
        features.append(feature)
        relations.append(list(graphs))
        names.append(name)
    return features, relations, names


def read_arrays(source, *, filename=None, delimiter=",", skip_header="auto", variable_names=None):
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
            raw = loadmat(stream, variable_names=variable_names)
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


def dataset_from_arrays(arrays, *, view_keys=None, views=None, graph_keys=None, label_key=None, label_column=None,
                        id_keys=None, label_id_key=None, mask_key=None,
                        transpose_views=(), detect_nan_rows=False, modalities=None, name="uploaded"):
    """Build a dataset using explicit array roles, shared by all file formats.

    ``views={"name": ("X1", ["A1", "A2"])}`` can reuse a feature in
    several views with different relations. A string value selects features
    only. With neither views nor view_keys, X*/A*/labels names are recognized.
    Explicit view_keys retain the existing opt-in label behavior.

    graph_keys maps selected view keys to lists of relation keys. With id_keys,
    graphs follow each view's local sample order. Without IDs the user asserts
    that every selected matrix has the same row order. Fully NaN rows may be
    detected only when explicitly enabled; partial NaNs are rejected. An
    explicit mask uses global row order and cannot be combined with ID alignment.

    ``label_column=(view_key, column_index)`` extracts labels from one feature
    array and removes that column before fitting. The index is zero-based,
    after any requested transpose. It cannot be combined with ``label_key``.
    """
    automatic = views is None and view_keys is None
    if views is not None and (view_keys is not None or graph_keys is not None):
        raise ValueError("Choose views or view_keys/graph_keys, not both")
    if automatic:
        views = default_views(arrays)
    if automatic and label_key is None:
        label_keys = array_roles(arrays)["labels"]
        if len(label_keys) > 1:
            raise ValueError("Multiple labels arrays found; select label_key explicitly")
        label_key = label_keys[0] if label_keys else None
    if views is not None:
        view_keys, relation_keys, view_names = _selection(arrays, views)
    else:
        if not view_keys or len(set(view_keys)) != len(view_keys):
            raise ValueError("Select at least one unique view key")
        graph_keys = graph_keys or {}
        if set(graph_keys) - set(view_keys):
            raise ValueError("Every graph must be mapped to a selected view")
        relation_keys = [graph_keys.get(key, []) for key in view_keys]
        view_names = list(view_keys)
    missing = set(view_keys) - set(arrays)
    if missing or (label_key is not None and label_key not in arrays):
        raise ValueError(f"Unknown selected array: {sorted(missing) or label_key}")
    if label_key in view_keys:
        raise ValueError("A labels array cannot also be selected as a feature view")
    if any(label_key in keys for keys in relation_keys):
        raise ValueError("A labels array cannot also be selected as a graph")
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
        if view_keys.count(key) != 1:
            raise ValueError("Extract embedded labels before reusing that feature in multiple views")
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
        labels = labels.toarray() if sparse.issparse(labels) else np.asarray(labels)
        if labels.ndim not in {1, 2} or (labels.ndim == 2 and 1 not in labels.shape):
            raise ValueError("Labels must be a vector with one value per sample")
    Xs = [x if sparse.issparse(x) else np.asarray(x) for x in Xs]
    graph_cache = {}
    for keys in relation_keys:
        for key in keys:
            if key not in arrays:
                raise ValueError(f"Unknown graph array: {key!r}")
            if key not in graph_cache:
                graph_cache[key] = sparse.csr_matrix(arrays[key])
    As = [None if not keys else graph_cache[keys[0]] if len(keys) == 1
          else [graph_cache[k] for k in keys] for keys in relation_keys]
    mapping = {name: (feature, keys) for name, feature, keys in zip(view_names, view_keys, relation_keys)}
    roles = array_roles(arrays)
    options = {"name": name, "view_names": view_names, "modalities": modalities,
               "metadata": {"available_features": roles["features"], "available_graphs": roles["graphs"],
                            "source_arrays": {k: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                              for k, value in arrays.items()},
                            "view_selection": mapping,
                            "import": {"view_keys": list(view_keys), "graph_keys": graph_keys, "views": mapping,
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


def load_mat(path, *, views=None, **mapping):
    """Load standardized MATLAB X*/A*/labels arrays, optionally composing views.

    No Hugging Face client or MATLAB installation is needed. For a different
    naming scheme pass explicit view_keys, graph_keys and label_key instead.
    """
    arrays = read_arrays(path)
    if views is not None and "label_key" not in mapping:
        label_keys = array_roles(arrays)["labels"]
        if len(label_keys) > 1:
            raise ValueError("Multiple labels arrays found; select label_key explicitly")
        mapping["label_key"] = label_keys[0] if label_keys else None
    return dataset_from_arrays(arrays, views=views, **mapping)


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
