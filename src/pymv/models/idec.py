"""PyTorch IDEC objective, with explicit early fusion for multiple views."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from pymv.progress import iterations

from ._common import apply_scalers, batches, fit_scalers, integer, number, optimize, validate_views
from ._deep import Autoencoder, DeepModel, parameter_budget
from ._torch_utils import initialization_seed
from .registry import register_model


def student_assignments(z, centers, alpha=1.):
    distances = (z[:, None, :] - centers[None, :, :]).square().sum(-1)
    q = (1 + distances / alpha).pow(-(alpha + 1) / 2)
    return q / q.sum(1, keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)


def target_distribution(q):
    weight = q.square() / q.sum(0, keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)
    return weight / weight.sum(1, keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)


@register_model("idec", description="Deep embedded clustering with reconstruction; early fusion for multiple views.",
                paper="https://www.ijcai.org/proceedings/2017/243",
                implementation="https://github.com/XifengGuo/IDEC")
class IDEC(DeepModel):
    """Autoencoder pretraining followed by reconstruction + gamma * KL(P || Q).

    ``pretrain_epochs`` counts full reconstruction epochs. ``max_iter`` and
    ``update_interval`` count refinement minibatch updates, as in IDEC.
    Multiple views are concatenated: this is early-fusion IDEC, not a new
    multiview objective. No labels are used in fitting or stopping.
    """

    def __init__(self, n_clusters, *, embedding_dim=10, hidden_dims=(500, 500, 2000),
                 pretrain_epochs=20, max_iter=1000, update_interval=20, gamma=.1, alpha=1.,
                 tol=.001, learning_rate=.001, batch_size=256, n_init=20,
                 preprocess=True, random_state=None, device="auto", max_memory_mb=1024):
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.pretrain_epochs = pretrain_epochs
        self.max_iter = max_iter
        self.update_interval = update_interval
        self.gamma = gamma
        self.alpha = alpha
        self.tol = tol
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_init = n_init
        self.preprocess = preprocess
        self.random_state = random_state
        self.device = device
        self.max_memory_mb = max_memory_mb

    def fit(self, Xs, As=None, *, observed_mask=None):
        xs, mask = validate_views(Xs, observed_mask, max_memory_mb=self.max_memory_mb)
        self._check_deep(len(xs[0]))
        integer("pretrain_epochs", self.pretrain_epochs)
        integer("update_interval", self.update_interval)
        number("gamma", self.gamma)
        number("alpha", self.alpha, positive=True)
        number("tol", self.tol)
        dims = [sum(x.shape[1] for x in xs), *self.hidden_dims, self.embedding_dim]
        parameter_budget([dims, dims[::-1]], self.max_memory_mb)
        scalers = fit_scalers(xs, mask, self.preprocess)
        fused = np.concatenate(apply_scalers(xs, mask, scalers), axis=1)
        self._execute(lambda device: self._train(fused, device))
        self.scalers_ = scalers
        self.n_features_in_ = [x.shape[1] for x in xs]
        self.embedding_dim_ = self.embedding_dim
        return self

    def _train(self, fused, device):
        rng = np.random.default_rng(self.random_state)
        with initialization_seed(self.random_state):
            model = Autoencoder(fused.shape[1], self.hidden_dims, self.embedding_dim)
            for layer in model.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        pretrain = []
        for _ in iterations(self.pretrain_epochs, "Autoencoder pretraining"):
            model.train()
            values = []
            for idx in batches(len(fused), self.batch_size, rng):
                x = torch.as_tensor(fused[idx], device=device)
                _, decoded = model(x)
                values.append(optimize(F.mse_loss(decoded, x), optimizer))
            pretrain.append(np.mean(values))
        def encode(m, xs):
            return m.encoder(xs[0])
        embedding = self._encode_batches(model, [fused], device, encode)
        km = KMeans(self.n_clusters, n_init=self.n_init, random_state=self.random_state).fit(embedding)
        model.register_parameter("cluster_centers", nn.Parameter(torch.as_tensor(km.cluster_centers_, device=device)))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        previous = km.labels_
        history, components = [], []
        batch_indices = list(batches(len(fused), self.batch_size))
        for step in iterations(self.max_iter, "Clustering minibatches"):
            if step % self.update_interval == 0:
                embedding = self._encode_batches(model, [fused], device, encode)
                with torch.no_grad():
                    q = student_assignments(torch.from_numpy(embedding),
                                            model.cluster_centers.detach().cpu(), self.alpha)
                    target = target_distribution(q).numpy()
                labels = q.argmax(1).numpy()
                changed = np.mean(labels != previous)
                previous = labels
                if step > 0 and changed < self.tol:
                    break
            model.train()
            idx = batch_indices[step % len(batch_indices)]
            x = torch.as_tensor(fused[idx], device=device)
            z, decoded = model(x)
            q = student_assignments(z, model.cluster_centers, self.alpha)
            kl = F.kl_div(q.clamp_min(1e-12).log(), torch.as_tensor(target[idx], device=device), reduction="batchmean")
            reconstruction = F.mse_loss(decoded, x)
            loss = reconstruction + self.gamma * kl
            history.append(optimize(loss, optimizer))
            components.append([float(reconstruction.detach()), float(kl.detach())])
        embedding = self._encode_batches(model, [fused], device, encode)
        centers = model.cluster_centers.detach().cpu().numpy().copy()
        labels = student_assignments(torch.from_numpy(embedding), torch.from_numpy(centers),
                                     self.alpha).argmax(1).numpy()
        return dict(model_=model.cpu().eval(), embedding_=embedding, centroids_=centers, labels_=labels,
                    pretrain_loss_history_=np.asarray(pretrain), loss_history_=np.asarray(history),
                    loss_components_=np.asarray(components), loss_component_names_=["reconstruction", "kl"],
                    n_iter_=len(history),
                    inertia_=float(np.square(embedding - centers[labels]).sum()))

    def transform(self, Xs, As=None):
        xs, _ = self._prediction_views(Xs)
        return self._encode_batches(self.model_, [np.concatenate(xs, axis=1)], torch.device("cpu"),
                                    lambda m, views: m.encoder(views[0]))

    def predict(self, Xs, As=None):
        z = torch.from_numpy(self.transform(Xs))
        return student_assignments(z, torch.from_numpy(self.centroids_), self.alpha).argmax(1).numpy()
