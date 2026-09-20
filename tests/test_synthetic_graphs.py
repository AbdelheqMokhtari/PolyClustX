from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from pymv.display.synthetic import SyntheticGraphSpec, SyntheticSpec, generate_synthetic


@pytest.mark.parametrize("kind", ["knn", "mutual_knn", "rbf_knn"])
@pytest.mark.parametrize("views", [1, 2, 4])
def test_synthetic_graph_types(kind, views):
    spec = SyntheticSpec(n_samples=30, dimensions=[4] * views, distributions=["Gaussian"] * views,
                         graph_specs=[SyntheticGraphSpec(kind, 4, 3.0)])
    data = generate_synthetic(spec)
    for relations in data.As:
        graph = relations[0]
        assert sparse.isspmatrix_csr(graph) and graph.shape == (30, 30)
        assert (graph != graph.T).nnz == 0
        assert np.all(graph.diagonal() == 0)
        assert graph.nnz <= 30 * 4 * 2
        assert np.isfinite(graph.data).all() and (graph.data > 0).all()
        assert (graph.data <= 1).all()
        if kind == "rbf_knn":
            assert np.any(graph.data < 1)
        else:
            assert (graph.data == 1).all()
    regenerated = generate_synthetic(spec)
    assert all((a[0] != b[0]).nnz == 0 for a, b in zip(data.As, regenerated.As))
    no_graphs = generate_synthetic(replace(spec, graph_specs=[]))
    assert all(np.array_equal(a, b) for a, b in zip(data.Xs, no_graphs.Xs))
    unlabeled = generate_synthetic(replace(spec, labeled=False))
    assert unlabeled.labels is None
    assert all((a[0] != b[0]).nnz == 0 for a, b in zip(data.As, unlabeled.As))


def test_multiple_shared_relations_and_fusion():
    spec = SyntheticSpec(n_samples=30, dimensions=[4, 5, 6], distributions=["Gaussian"] * 3,
                         graph_specs=[SyntheticGraphSpec("knn", 3), SyntheticGraphSpec("rbf_knn", 6, 2)],
                         graph_views=[0, 2], shared_graph_view=1)
    data = generate_synthetic(spec)
    assert data.As[1] is None
    assert len(data.As[0]) == len(data.As[2]) == 2
    for left, right in zip(data.As[0], data.As[2]):
        assert (left != right).nnz == 0
        assert not np.shares_memory(left.data, right.data)
    assert data.metadata["synthetic"]["graph_specs"][1]["kind"] == "rbf_knn"
    for method in ["sum", "hadamard"]:
        fused = data.fused_graphs(method=method)
        assert fused[0].shape == (30, 30) and fused[1] is None


def test_mutual_graph_is_subset_of_union():
    base = SyntheticSpec(n_samples=30, graph_specs=[SyntheticGraphSpec("knn", 3)])
    union = generate_synthetic(base).As[0][0]
    mutual = generate_synthetic(replace(base, graph_specs=[SyntheticGraphSpec("mutual_knn", 3)])).As[0][0]
    assert mutual.nnz <= union.nnz
    assert mutual.multiply(union).nnz == mutual.nnz


@pytest.mark.parametrize("changes, message", [
    ({"graph_specs": [SyntheticGraphSpec("invalid")]}, "Graph kind"),
    ({"graph_specs": [SyntheticGraphSpec(neighbors=-1)]}, "neighbors"),
    ({"graph_specs": [SyntheticGraphSpec(neighbors=30)]}, "neighbors"),
    ({"graph_specs": [SyntheticGraphSpec(bandwidth=0)]}, "bandwidth"),
    ({"graph_specs": [SyntheticGraphSpec(bandwidth=float("nan"))]}, "bandwidth"),
    ({"graph_views": []}, "target view"),
    ({"graph_views": [0, 0]}, "unique"),
    ({"shared_graph_view": 2}, "indices"),
    ({"graph_relations": -1}, "non-negative"),
    ({"n_samples": 100000, "graph_specs": [SyntheticGraphSpec(neighbors=50)]}, "budget"),
])
def test_graph_validation(changes, message):
    spec = SyntheticSpec(n_samples=30, graph_specs=[SyntheticGraphSpec(neighbors=3)])
    with pytest.raises(ValueError, match=message):
        generate_synthetic(replace(spec, **changes))

