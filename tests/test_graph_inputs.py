import io

import numpy as np
import pytest
from scipy import sparse

from pymv.data import MultiViewDataset
from pymv.display.graph_uploads import attach_graphs, read_graph_file
from pymv.display.workspace import Workspace


def test_graph_objects_follow_sample_ids_and_parallel_edges():
    nx = pytest.importorskip("networkx")
    graph = nx.MultiGraph()
    graph.add_nodes_from(["b", "a", "c"])
    graph.add_edge("a", "b", weight=2)
    graph.add_edge("a", "b", weight=3)
    graph.add_edge("c", "c", weight=4)
    data = MultiViewDataset([np.ones((3, 2))], As=[[graph, np.eye(3)]], sample_ids=["a", "b", "c"])
    np.testing.assert_array_equal(data.As[0][0].toarray(), [[0, 5, 0], [5, 0, 0], [0, 0, 4]])
    assert data.fused_graphs()[0].shape == (3, 3)
    with pytest.raises(ValueError, match="node IDs"):
        MultiViewDataset([np.ones((3, 2))], As=[graph])


def test_graphml_and_sparse_npz_import():
    nx = pytest.importorskip("networkx")
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(2, 0, weight=2)
    output = io.BytesIO()
    nx.write_graphml(graph, output)
    result = read_graph_file(output.getvalue(), "graph.graphml", np.arange(3))
    assert result[2, 0] == 2 and result[0, 2] == 0
    with pytest.raises(ValueError, match="node IDs"):
        read_graph_file(output.getvalue(), "graph.graphml", [1, 2, 3])
    output = io.BytesIO()
    sparse.save_npz(output, result)
    np.testing.assert_array_equal(read_graph_file(output.getvalue(), "graph.npz", [0, 1, 2]).toarray(),
                                  result.toarray())


def test_graph_attachments_reload_without_duplication_or_source_mutation():
    ws = Workspace()
    source = MultiViewDataset([np.ones((3, 2))])
    key = ws.add_dataset(source)
    original = ws.load_dataset(key)
    attach_graphs(ws, key, 0, [sparse.eye(3)])
    assert original.As == [None] and source.As == [None]
    assert len(ws.load_dataset(key).As[0]) == 1
    ws.unload_dataset(key)
    assert len(ws.load_dataset(key).As[0]) == 1
    attach_graphs(ws, key, 0, [sparse.eye(3) * 2])
    ws.unload_dataset(key)
    assert len(ws.load_dataset(key).As[0]) == 2
    ws.remove_dataset(key)


def test_invalid_attachment_does_not_change_source():
    ws = Workspace()
    key = ws.add_dataset(MultiViewDataset([np.ones((3, 2))]))
    with pytest.raises(ValueError, match="Graphs must"):
        attach_graphs(ws, key, 0, [sparse.eye(4)])
    assert ws.load_dataset(key).As == [None]
