from types import SimpleNamespace

import pytest

from polyclustx.progress import iterations, terminal_training_progress, training_progress


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


def test_terminal_progress_keeps_existing_observer_and_restores_it():
    events, labels = [], []
    bar = SimpleNamespace(set_postfix_str=labels.append)
    with training_progress(events.append):
        with terminal_training_progress(bar, "toy / model [run 1/2]"):
            assert list(iterations(3, "Training")) == [0, 1, 2]
        assert list(iterations(1, "After")) == [0]
    assert events[0]["stage"] == "Training" and events[-1]["stage"] == "After"
    assert "preparing and fitting" in labels[0]
    assert any("Training 3/3" in label and "stage ETA 00:00" in label for label in labels)
    assert labels[-1] == "toy / model [run 1/2]"
    count = len(events)
    list(iterations(1))
    assert len(events) == count


def test_terminal_progress_restores_observer_after_failure():
    events, labels = [], []
    with training_progress(events.append):
        with pytest.raises(RuntimeError):
            with terminal_training_progress(SimpleNamespace(set_postfix_str=labels.append), "test"):
                list(iterations(2, "Before"))
                raise RuntimeError("fit failed")
        list(iterations(1, "After"))
    assert events[-1]["stage"] == "After" and labels[-1] == "test"


def test_no_terminal_bar_preserves_training_observation():
    events = []
    with training_progress(events.append), terminal_training_progress(None, "quiet"):
        assert list(iterations(2)) == [0, 1]
    assert events[-1]["completed"] == 2


def test_interface_launcher_threads_and_clean_interrupt(monkeypatch):
    import subprocess

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
