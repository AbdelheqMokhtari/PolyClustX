"""Context-local training notifications, independent of any interface toolkit."""
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

_callback = ContextVar("polyclustx_progress", default=None)


@contextmanager
def training_progress(callback):
    """Observe fitting in this thread/task without changing model parameters."""
    token = _callback.set(callback)
    try:
        yield
    finally:
        _callback.reset(token)


@contextmanager
def terminal_training_progress(bar, label):
    """Show reported training stages on an existing terminal progress bar.

    Preserve an enclosing observer, including interface callbacks. Estimates
    cover the current stage only; preparing inputs and later stages are not
    included. Without a bar, observation behaves exactly as before.
    """
    if bar is None:
        yield
        return

    from tqdm import tqdm

    observer = _callback.get()

    def update(event):
        if observer is not None:
            observer(event)
        remaining = event["remaining"]
        eta = "pending" if remaining is None else tqdm.format_interval(remaining)
        bar.set_postfix_str(
            f"{label} | {event['stage']} {event['completed']}/{event['total']} | stage ETA {eta}"
        )

    bar.set_postfix_str(f"{label} | preparing and fitting")
    try:
        with training_progress(update):
            yield
    finally:
        bar.set_postfix_str(label)


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
