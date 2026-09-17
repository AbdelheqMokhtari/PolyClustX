import io

import numpy as np
import pytest
from scipy import sparse
from scipy.io import savemat

from pymv.data import (
    MultiViewDataset,
    align_views,
    dataset_from_arrays,
    impute_missing,
    read_arrays,
    simulate_missing_views,
)
from pymv.preprocessing import fuse_graphs


@pytest.mark.parametrize("views", [1, 2, 4])
def test_unlabeled_views(views):
    data = MultiViewDataset([np.ones((6, 3))] * views)
    assert data.n_views == views and data.labels is None and data.n_clusters is None
    assert data.is_complete and data.category == "multiview"


def test_alignment_graphs_and_labels():
    data = align_views([np.array([[1], [2]]), np.array([[30], [20]])],
                       [["a", "b"], ["c", "b"]], graphs=[[np.eye(2), np.ones((2, 2))], None],
                       labels=[2, 1, 0], label_ids=["c", "b", "a"], modalities=["image", "text"])
    assert data.sample_ids.tolist() == ["a", "b", "c"]
    assert data.labels.tolist() == [0, 1, 2]
    assert data.Xs[1].ravel().tolist() == [0, 20, 30]
    assert data.observed_mask.tolist() == [[True, False], [True, True], [False, True]]
    assert data.category == "attributed_graph_multimodal"
    assert data.As[0][0].shape == (3, 3)
    assert data.As[0][0][2, 2] == 0


def test_missing_simulation_and_imputation():
    original = MultiViewDataset([np.arange(40).reshape(10, 4), np.ones((10, 2))])
    data = simulate_missing_views(original, .2, random_state=7)
    assert (~data.observed_mask).sum() == 4
    assert data.observed_mask.any(axis=1).all()
    assert original.is_complete
    np.testing.assert_array_equal(data.observed_mask,
                                  simulate_missing_views(original, .2, random_state=7).observed_mask)
    complete = impute_missing(data)
    assert complete.is_complete and not data.is_complete
    for j, x in enumerate(complete.Xs):
        if (~data.observed_mask[:, j]).any():
            np.testing.assert_allclose(x[~data.observed_mask[:, j]][0],
                                       data.Xs[j][data.observed_mask[:, j]].mean(axis=0))
    with pytest.raises(ValueError, match="capacity"):
        simulate_missing_views(original, .8)
    with pytest.raises(ValueError, match="only view"):
        simulate_missing_views(MultiViewDataset([np.ones((10, 3))]), .2)


def test_missing_dispatch_never_passes_labels():
    class Unsupported:
        def fit(self, *args, **kwargs):
            raise AssertionError("Should reject before training")

    data = MultiViewDataset([np.ones((3, 2)), np.ones((3, 1))], labels=[1, 2, 1],
                            observed_mask=np.array([[1, 0], [1, 1], [0, 1]], dtype=bool))
    with pytest.raises(ValueError, match="does not support missing"):
        data.fit(Unsupported())

    class Supported:
        supports_missing_views = True

        def fit(self, Xs, As, *, observed_mask):
            assert observed_mask.shape == (3, 2)
            return self

    assert isinstance(data.fit(Supported()), Supported)


@pytest.mark.parametrize("as_sparse", [True, False])
def test_relation_fusion(as_sparse):
    a, b = np.array([[0, 2], [2, 0]]), np.array([[0, 3], [3, 0]])
    graphs = [sparse.csr_matrix(a), b] if as_sparse else [a, b]
    result = fuse_graphs(graphs, weights=[.25, .75])
    np.testing.assert_allclose(result.toarray() if as_sparse else result, .25 * a + .75 * b)
    result = fuse_graphs(graphs, method="hadamard")
    np.testing.assert_allclose(result.toarray() if as_sparse else result, a * b)
    with pytest.raises(ValueError):
        fuse_graphs(graphs, weights=[1, -1])
    with pytest.raises(ValueError):
        fuse_graphs(graphs, method="hadamard", weights=[1, 1])
    with pytest.raises(ValueError):
        fuse_graphs([a, np.eye(3)])


def test_import_formats_and_explicit_roles(tmp_path):
    x, y = np.ones((4, 2)), np.arange(4)
    buffer = io.BytesIO()
    np.savez(buffer, arbitrary=x, truth=y)
    arrays = read_arrays(buffer.getvalue(), filename="data.npz")
    data = dataset_from_arrays(arrays, view_keys=["arbitrary"])
    assert data.labels is None
    assert dataset_from_arrays(arrays, view_keys=["arbitrary"], label_key="truth").labels.tolist() == y.tolist()
    path = tmp_path / "views.mat"
    cells = np.empty((1, 2), dtype=object)
    cells[0, 0], cells[0, 1] = x, x[:, :1]
    savemat(path, {"features": cells, "y": y})
    assert set(read_arrays(path)) == {"features[0]", "features[1]", "y"}
    assert read_arrays(b"1,2\n3,4\n", filename="v.csv")["v"].shape == (2, 2)


def test_nan_detection_is_explicit_and_partial_nan_rejected():
    arrays = {"a": np.array([[1, 2], [np.nan, np.nan]]), "b": np.ones((2, 1))}
    with pytest.raises(ValueError, match="finite"):
        dataset_from_arrays(arrays, view_keys=["a", "b"])
    data = dataset_from_arrays(arrays, view_keys=["a", "b"], detect_nan_rows=True)
    assert not data.is_complete
    arrays["a"][1] = [np.nan, 1]
    with pytest.raises(ValueError, match="finite"):
        dataset_from_arrays(arrays, view_keys=["a", "b"], detect_nan_rows=True)


def test_validation():
    with pytest.raises(ValueError, match="unique IDs"):
        align_views([np.ones((2, 1))], [[1, 1]])
    with pytest.raises(ValueError, match="exactly one entry"):
        MultiViewDataset([np.ones((2, 2)), np.ones((2, 1))], [np.eye(2)])
    with pytest.raises(ValueError, match="at least one"):
        MultiViewDataset([np.ones((2, 2))], observed_mask=np.zeros((2, 1), dtype=bool))
