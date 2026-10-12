"""Contrastive fusion with divergence-based clustering (CONAN, BigData 2021).

Feature-matrix MLP route; see THIRD_PARTY_NOTICES.md for the reference MIT notice.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pymv.progress import iterations

from ._common import apply_scalers, batches, fit_scalers, memory_check, number, optimize, validate_views
from ._deep import DeepModel, mlp, parameter_budget
from ._torch_utils import initialization_seed
from .registry import register_model
from .scm import SCM


def divergence_loss(q, hidden):
    """DDC: kernel Cauchy–Schwarz separation, orthogonality and simplex loss."""
    n, k = q.shape
    if k == 1:
        return q.sum() * 0
    distance = torch.cdist(hidden, hidden).square()
    bandwidth = (.15 * distance.detach().median()).clamp_min(1e-9)
    kernel = torch.exp(-distance / (2 * bandwidth))

    def separation(assignments):
        gram = assignments.T @ kernel @ assignments
        scale = (gram.diag()[:, None] * gram.diag()[None, :]).clamp_min(1e-18).sqrt()
        return 2 * torch.triu(gram / scale, diagonal=1).sum() / (k * (k - 1))

    simplex = torch.exp(-torch.cdist(q, torch.eye(k, device=q.device, dtype=q.dtype)).square())
    orthogonality = 2 * torch.triu(q @ q.T, diagonal=1).sum() / max(n * (n - 1), 1)
    return separation(q) + orthogonality + separation(simplex)


class FusionNetwork(nn.Module):
    def __init__(self, widths, hidden, dim, projection, k):
        super().__init__()
        self.encoders = nn.ModuleList([mlp([width, *hidden, dim]) for width in widths])
        self.fusion = mlp([len(widths) * dim, dim, dim], batchnorm=True)
        self.projector = mlp([dim, projection, projection], batchnorm=True)
        self.cluster_hidden = mlp([dim, projection], batchnorm=True, activate_last=True)
        self.cluster_head = mlp([projection, k], softmax=True)

    def forward(self, xs):
        views = [encoder(x) for encoder, x in zip(self.encoders, xs)]
        fused = self.fusion(torch.cat(views, dim=1))
        hidden = self.cluster_hidden(fused)
        return views, fused, hidden, self.cluster_head(hidden)


@register_model("conan", description="Contrastive fusion networks with divergence-based clustering.",
                paper="https://doi.org/10.1109/BigData52589.2021.9671851",
                implementation="https://github.com/Guanzhou-Ke/conan")
class CONAN(DeepModel):
    """Learn a nonlinear fusion aligned with each view using NT-Xent and DDC.

    This implements the feature-MLP, SimCLR + DDC route. It does not wrap the
    reference image backbones or its optional DEC/SimSiam ablations. One view
    retains the encoder-to-fusion contrastive task. Graphs are not used.
    """

    n_init = 1

    def __init__(self, n_clusters, *, embedding_dim=128, hidden_dims=(256, 256), projection_dim=128,
                 max_iter=100, batch_size=128, learning_rate=.001, temperature=.1,
                 contrastive_weight=1., preprocess=True, random_state=None, device="auto", max_memory_mb=1024):
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.projection_dim = projection_dim
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.preprocess = preprocess
        self.random_state = random_state
        self.device = device
        self.max_memory_mb = max_memory_mb

    def fit(self, Xs, As=None, *, observed_mask=None):
        from ._common import integer
        xs, mask = validate_views(Xs, observed_mask, max_memory_mb=self.max_memory_mb)
        self._check_deep(len(xs[0]))
        integer("projection_dim", self.projection_dim)
        number("temperature", self.temperature, positive=True)
        number("contrastive_weight", self.contrastive_weight)
        paths = [[x.shape[1], *self.hidden_dims, self.embedding_dim] for x in xs]
        paths += [[len(xs) * self.embedding_dim, self.embedding_dim, self.embedding_dim],
                  [self.embedding_dim, self.projection_dim, self.projection_dim],
                  [self.embedding_dim, self.projection_dim, self.n_clusters]]
        parameter_budget(paths, self.max_memory_mb)
        memory_check(min(len(xs[0]), self.batch_size + 1)**2 * 4 * 32, self.max_memory_mb)
        scalers = fit_scalers(xs, mask, self.preprocess)
        self._execute(lambda device: self._train(apply_scalers(xs, mask, scalers), device))
        self.scalers_, self.n_features_in_ = scalers, [x.shape[1] for x in xs]
        self.embedding_dim_ = self.embedding_dim
        return self

    def _train(self, xs, device):
        with initialization_seed(self.random_state):
            model = FusionNetwork([x.shape[1] for x in xs], self.hidden_dims, self.embedding_dim,
                                  self.projection_dim, self.n_clusters).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        rng = np.random.default_rng(self.random_state)
        history, components = [], []
        for _ in iterations(self.max_iter, "Contrastive fusion epochs"):
            model.train()
            values, terms = [], []
            for idx in batches(len(xs[0]), self.batch_size, rng):
                views, fused, hidden, q = model([torch.as_tensor(x[idx], device=device) for x in xs])
                clustering = divergence_loss(q, hidden)
                projected = model.projector(fused)
                # Stack positive pairs by sample, not by feature as in the reference's SimCLR bug.
                contrastive = sum(SCM._contrastive_loss(projected, model.projector(v), self.temperature)
                                  for v in views)
                values.append(optimize(clustering + self.contrastive_weight * contrastive, optimizer))
                terms.append([float(clustering.detach()), float(contrastive.detach())])
            history.append(np.mean(values))
            components.append(np.mean(terms, axis=0))
        embedding = self._encode_batches(model, xs, device, lambda m, x: m(x)[1])
        probabilities = self._encode_batches(model, xs, device, lambda m, x: m(x)[3])
        return dict(model_=model.cpu(), embedding_=embedding, labels_=probabilities.argmax(1),
                    probabilities_=probabilities, loss_history_=np.asarray(history),
                    loss_components_=np.asarray(components), loss_component_names_=["ddc", "contrastive"])

    def transform(self, Xs):
        xs, _ = self._prediction_views(Xs)
        return self._encode_batches(self.model_, xs, "cpu", lambda m, x: m(x)[1])

    def predict(self, Xs):
        xs, _ = self._prediction_views(Xs)
        return self._encode_batches(self.model_, xs, "cpu", lambda m, x: m(x)[3]).argmax(1)
