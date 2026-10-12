"""Context-local training notifications, independent of any interface toolkit."""
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

_callback = ContextVar("pymv_progress", default=None)


@contextmanager
def training_progress(callback):
    """Observe fitting in this thread/task without changing model parameters."""
    token = _callback.set(callback)
    try:
        yield
    finally:
        _callback.reset(token)


def iterations(count, stage="Training"):
    """Yield iteration indices and estimate remaining stage time from elapsed work.

    Estimates exclude future stages and can change with early convergence. No
    training iterations are removed or capped by progress reporting.
    """
    callback = _callback.get()
    start = last = perf_counter()
    if callback:
        callback({"stage": stage, "completed": 0, "total": count, "elapsed": 0., "remaining": None})
    for index in range(count):
        yield index
        now = perf_counter()
        if callback and (now - last >= .25 or index + 1 == count):
            elapsed = now - start
            callback({"stage": stage, "completed": index + 1, "total": count, "elapsed": elapsed,
                      "remaining": elapsed / (index + 1) * (count - index - 1)})
            last = now
