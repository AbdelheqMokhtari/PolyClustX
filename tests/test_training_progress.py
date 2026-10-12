from polyclustx.progress import iterations, training_progress


def test_callbacks_are_scoped_and_do_not_skip_iterations():
    events = []
    with training_progress(events.append):
        assert list(iterations(3, "Test")) == [0, 1, 2]
    assert events[0]["remaining"] is None
    assert events[-1]["completed"] == 3
    assert events[-1]["remaining"] == 0
    count = len(events)
    list(iterations(2))
    assert len(events) == count


def test_model_emits_training_events():
    import numpy as np

    from polyclustx.models import IDEC
    events = []
    with training_progress(events.append):
        IDEC(2, hidden_dims=(4,), embedding_dim=2, max_iter=2, pretrain_epochs=1,
             n_init=1, batch_size=8, device="cpu").fit([np.random.default_rng(1).normal(size=(12, 3))])
    assert {e["stage"] for e in events} == {"Autoencoder pretraining", "Clustering minibatches"}


def test_interface_launcher_threads_and_clean_interrupt(monkeypatch):
    import subprocess

    import pytest

    from polyclustx.display import launch

    monkeypatch.setenv("POLYCLUSTX_CPU_THREADS", "3")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    calls = []

    def interrupted(command, **kwargs):
        calls.append((command, kwargs))
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "call", interrupted)
    with pytest.raises(SystemExit) as stopped:
        launch(["--server.port=8519"])
    assert stopped.value.code == 0
    assert calls[0][1]["env"]["OMP_NUM_THREADS"] == "4"
    assert calls[0][1]["env"]["MKL_NUM_THREADS"] == "3"
    assert calls[0][0][-1] == "--server.port=8519"
