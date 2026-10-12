"""Validation and bounded PyTorch execution shared by clustering estimators."""
from __future__ import annotations

import logging

import numpy as np
import torch
from scipy import sparse
from sklearn.exceptions import NotFittedError

from pymv.core import BaseModel
from pymv.models._torch_utils import resolve_device


def integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def number(name, value, minimum=0, positive=False):
    if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < minimum or (positive and value <= minimum):
        raise ValueError(f"{name} must be {'>' if positive else '>='} {minimum}")


def memory_check(byte_count, limit):
    number("max_memory_mb", limit, positive=True)
    if byte_count > limit * 1024**2:
        raise ValueError(f"Estimated allocation exceeds max_memory_mb={limit}; reduce samples, "
                         "network widths or views, or explicitly increase the budget")


def validate_views(Xs, observed_mask=None, *, missing=False, dtype=np.float32, max_memory_mb=1024):
    if not isinstance(Xs, (list, tuple)) or not Xs:
        raise ValueError("Xs must be a non-empty list of aligned feature matrices")
    shapes = [getattr(x, "shape", ()) for x in Xs]
    if any(len(s) != 2 or min(s) < 1 or s[0] != shapes[0][0] for s in shapes):
        raise ValueError("Views must be nonempty 2D matrices with the same sample count")
    memory_check(sum(s[0] * s[1] for s in shapes) * np.dtype(dtype).itemsize * 3, max_memory_mb)
    shape = (shapes[0][0], len(Xs))
    mask = np.ones(shape, dtype=bool) if observed_mask is None else np.asarray(observed_mask)
    if mask.shape != shape or mask.dtype != bool or not mask.any(axis=1).all():
        raise ValueError("observed_mask must be boolean (samples, views), with a view present for every sample")
    if not missing and not mask.all():
        raise ValueError("This model requires complete views; explicitly impute missing views first")
    arrays = []
    for v, x in enumerate(Xs):
        a = np.array(x.toarray() if sparse.issparse(x) else x, dtype=dtype, copy=True)
        if not np.isfinite(a[mask[:, v]]).all():
            raise ValueError("Observed features must be finite")
        a[~mask[:, v]] = 0
        arrays.append(a)
    return arrays, mask.copy()


class TorchModel(BaseModel):
    """CPU NumPy outputs, bounded allocations and explicit CPU/CUDA policy."""

    backend = "pytorch"

    def fit_predict(self, Xs, As=None, *, observed_mask=None):
        return self.fit(Xs, As, observed_mask=observed_mask).labels_

    def _check_common(self, n_samples):
        integer("n_clusters", self.n_clusters)
        integer("max_iter", self.max_iter)
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed the number of samples")
        if self.random_state is not None:
            integer("random_state", self.random_state, 0)
            if self.random_state >= 2**32:
                raise ValueError("random_state must be smaller than 2**32")

    def _execute(self, operation):
        device = resolve_device(self.device)
        result = None
        try:
            result = operation(device)
        except torch.cuda.OutOfMemoryError:
            if self.device != "auto" or device.type != "cuda":
                raise
            logging.getLogger(__name__).warning("CUDA out of memory; restarting %s on CPU", type(self).__name__)
        if result is None:
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            result = operation(device)
        self.__dict__.update(result)
        self.device_ = str(device)
        return self

    def _prediction_views(self, Xs, observed_mask=None):
        if not hasattr(self, "n_features_in_"):
            raise NotFittedError("Call fit before transform or predict")
        xs, mask = validate_views(Xs, observed_mask, missing=self.supports_missing_views,
                                  max_memory_mb=self.max_memory_mb)
        if [x.shape[1] for x in xs] != self.n_features_in_:
            raise ValueError("Prediction needs the same view order and feature dimensions as fitting")
        if hasattr(self, "scalers_"):
            xs = apply_scalers(xs, mask, self.scalers_)
        return xs, mask


def fit_scalers(xs, mask, preprocess):
    scalers = []
    for v, x in enumerate(xs):
        present = x[mask[:, v]]
        if not len(present):
            raise ValueError(f"View {v} has no observed training samples")
        low = present.min(axis=0) if preprocess else np.zeros(x.shape[1], dtype=x.dtype)
        span = present.max(axis=0) - low if preprocess else np.ones(x.shape[1], dtype=x.dtype)
        scalers.append((low, np.where(span > 0, span, 1)))
    return scalers


def apply_scalers(xs, mask, scalers):
    result = []
    for v, (x, (low, span)) in enumerate(zip(xs, scalers)):
        a = (x - low) / span
        a[~mask[:, v]] = 0
        if not np.isfinite(a).all():
            raise ValueError("Feature scaling overflowed; rescale the input features")
        result.append(a)
    return result


def batches(n, batch_size, rng=None):
    indices = np.arange(n) if rng is None else rng.permutation(n)
    # Merge a trailing singleton into the preceding batch for BatchNorm.
    for start in range(0, n, batch_size):
        if start > 0 and n - start == 1:
            break
        stop = min(start + batch_size, n)
        if n - stop == 1:
            stop = n
        yield indices[start:stop]


def optimize(loss, optimizer):
    if not torch.isfinite(loss):
        raise ValueError("Training loss became nonfinite; check feature scale and learning rate")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all()
           for group in optimizer.param_groups for p in group["params"]):
        raise ValueError("Training gradients became nonfinite; reduce the learning rate")
    optimizer.step()
    return float(loss.detach())
