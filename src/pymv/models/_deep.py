"""Small neural building blocks; all fitting state stays inside the estimator."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ._common import TorchModel, batches, integer, memory_check, number


def linear(left, right):
    return nn.Linear(left, right, device="cpu", dtype=torch.float32)


def mlp(widths, *, batchnorm=False, activate_last=False, softmax=False):
    layers = []
    for i, (left, right) in enumerate(zip(widths, widths[1:])):
        layers.append(linear(left, right))
        if i < len(widths) - 2 or activate_last:
            if batchnorm:
                layers.append(nn.BatchNorm1d(right, device="cpu", dtype=torch.float32))
            layers.append(nn.ReLU())
    if softmax:
        layers.append(nn.Softmax(dim=1))
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    def __init__(self, width, hidden, latent, *, probability=False, batchnorm=False):
        super().__init__()
        dims = [width, *hidden, latent]
        self.encoder = mlp(dims, batchnorm=batchnorm, softmax=probability)
        self.decoder = mlp(dims[::-1], batchnorm=batchnorm, activate_last=probability)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def parameter_budget(paths, max_memory_mb):
    # Weights, gradients, Adam moments and temporary optimizer buffers.
    count = sum((left + 1) * right for widths in paths for left, right in zip(widths, widths[1:]))
    memory_check(count * 24, max_memory_mb)


class DeepModel(TorchModel):
    def _check_deep(self, n):
        self._check_common(n)
        integer("embedding_dim", self.embedding_dim)
        integer("batch_size", self.batch_size, 2)
        integer("n_init", self.n_init)
        number("learning_rate", self.learning_rate, positive=True)
        if n < 2:
            raise ValueError("Neural models require at least two training samples")
        if not isinstance(self.hidden_dims, (tuple, list)) or not self.hidden_dims:
            raise ValueError("hidden_dims must contain positive layer widths")
        for width in self.hidden_dims:
            integer("hidden_dims", width)

    @torch.no_grad()
    def _encode_batches(self, model, xs, device, function):
        model.eval()
        output = []
        for indices in batches(len(xs[0]), self.batch_size):
            tensors = [torch.as_tensor(x[indices], device=device) for x in xs]
            output.append(function(model, tensors).cpu().numpy())
        return np.concatenate(output)
