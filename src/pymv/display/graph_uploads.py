"""Explicit graph-file imports and disk-backed attachments for session datasets."""

from __future__ import annotations

import copy
import io
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy import sparse

from pymv.data import read_arrays
from pymv.data.graphs import graph_adjacency


def read_graph_file(content, filename, sample_ids):
    """Read an adjacency array or GraphML/GEXF graph with matching sample IDs."""
    suffix = Path(filename).suffix.lower()
    n = len(sample_ids)
    if suffix in {".graphml", ".gexf"}:
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError('Repair the installation with: pip install --upgrade pymv-clustering') from exc
        reader = nx.read_graphml if suffix == ".graphml" else nx.read_gexf
        graph = reader(io.BytesIO(content))
        # XML graph formats store node identifiers as strings. Resolve only exact
        # string representations, not a positional or numeric guess.
        ids = {str(value): value for value in sample_ids}
        if len(ids) != n or set(graph.nodes) != set(ids):
            raise ValueError("Graph node IDs must match sample IDs, including isolated samples")
        return graph_adjacency(nx.relabel_nodes(graph, ids), sample_ids)
    adjacency = None
    if suffix == ".npz":
        try:
            adjacency = sparse.load_npz(io.BytesIO(content))
        except (ValueError, KeyError):
            pass
    if adjacency is None:
        arrays = read_arrays(content, filename=filename)
        if len(arrays) != 1:
            raise ValueError("Upload one adjacency per file; use the container importer to map multiple arrays")
        adjacency = next(iter(arrays.values()))
    adjacency = sparse.csr_matrix(adjacency, dtype=float)
    if adjacency.shape != (n, n) or not np.isfinite(adjacency.data).all() or (adjacency.data < 0).any():
        raise ValueError(f"Adjacency must be a finite nonnegative {n} × {n} matrix in sample order")
    return adjacency


def attach_graphs(workspace, dataset_id, view, graphs):
    """Append relations to one view, staging them on disk for lazy reloads."""
    resource = workspace.datasets[dataset_id]
    data = workspace.load_dataset(dataset_id)
    if not isinstance(view, int) or not 0 <= view < data.n_views or not graphs:
        raise ValueError("Choose an existing view and at least one graph")
    directory = Path(tempfile.mkdtemp(prefix="pymv-graphs-"))
    previous_loader, previous_cleanup = resource.loader, resource.cleanup
    try:
        paths = []
        for index, graph in enumerate(graphs):
            path = directory / f"relation-{index}.npz"
            sparse.save_npz(path, sparse.csr_matrix(graph))
            paths.append(path)

        def add_relations(dataset):
            current = dataset.As[view]
            relations = [] if current is None else list(current) if isinstance(current, (list, tuple)) else [current]
            dataset.As[view] = relations + [sparse.load_npz(path) for path in paths]
            return dataset

        # Validate shapes/weights before installing a loader or changing the session.
        candidate = copy.copy(data)
        candidate.As = list(data.As)
        add_relations(candidate)
        from pymv.data import MultiViewDataset
        MultiViewDataset(data.Xs, candidate.As, sample_ids=data.sample_ids, observed_mask=data.observed_mask)

        def loader():
            return add_relations(previous_loader())

        def cleanup():
            shutil.rmtree(directory)
            if previous_cleanup:
                previous_cleanup()

        resource.loader = loader
        resource.cleanup = cleanup
        resource._data = candidate
    except Exception:
        shutil.rmtree(directory)
        raise
