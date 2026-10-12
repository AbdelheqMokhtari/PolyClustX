from pymv.progress import iterations, training_progress


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

    from pymv.models import IDEC
    events = []
    with training_progress(events.append):
        IDEC(2, hidden_dims=(4,), embedding_dim=2, max_iter=2, pretrain_epochs=1,
             n_init=1, batch_size=8, device="cpu").fit([np.random.default_rng(1).normal(size=(12, 3))])
    assert {e["stage"] for e in events} == {"Autoencoder pretraining", "Clustering minibatches"}
