from polyclustx.display.workspace import Workspace
from tests.test_display_comparisons import app_test


def test_synthetic_graph_controls_and_lazy_reload():
    workspace = Workspace()
    at = app_test(workspace, 0)
    next(r for r in at.radio if r.label == "Data source").set_value("Synthetic").run()
    next(n for n in at.number_input if n.label == "Samples").set_value(30).run()
    next(s for s in at.selectbox if s.label == "Graph generation").select("Shared graphs").run()
    next(n for n in at.number_input if n.label == "Relations per selected view").set_value(2).run()
    at.selectbox(key="synthetic_graph_kind_1").select("Weighted kNN (RBF)").run()
    next(b for b in at.button if b.label == "Create synthetic dataset").click().run()
    assert not at.exception and not at.error
    resource = next(iter(workspace.datasets.values()))
    data = resource.load()
    assert len(data.As[0]) == len(data.As[1]) == 2
    assert (data.As[0][1] != data.As[1][1]).nnz == 0
    resource.unload()
    assert (resource.load().As[0][1] != data.As[0][1]).nnz == 0
    next(b for b in at.button if b.label == "Apply changes").click().run()
    assert not at.exception and not at.error and not workspace.pending_datasets
