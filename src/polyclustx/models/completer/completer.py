"""Contrastive prediction with observed-mask-aware latent completion.

COMPLETER components are adapted from the MIT-licensed XLearning reference;
see LICENSE for third-party notices. Framework fitting does not consume class labels.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from polyclustx.progress import iterations

from .._common import apply_scalers, batches, fit_scalers, integer, number, optimize, validate_views
from .._deep import Autoencoder, DeepModel, mlp, parameter_budget
from .._torch_utils import initialization_seed
from ..registry import register_model


def consistency_loss(left, right, marginal_weight):
    """Symmetric empirical joint-distribution objective, without a B x d x d allocation."""
    joint = left.T @ right
    joint = (joint + joint.T) / 2
    joint = joint / joint.sum().clamp_min(1e-15)
    row = joint.sum(1, keepdim=True).clamp_min(1e-15)
    column = joint.sum(0, keepdim=True).clamp_min(1e-15)
    return -(joint * (joint.clamp_min(1e-15).log()
                      - marginal_weight * row.log() - marginal_weight * column.log())).sum()


class PredictionNetwork(nn.Module):
    def __init__(self, widths, hidden, latent, predictor_hidden, batchnorm):
        super().__init__()
        self.autoencoders = nn.ModuleList([Autoencoder(w, hidden, latent, probability=True, batchnorm=batchnorm)
                                          for w in widths])
        path = [latent, *predictor_hidden]
        self.predictors = nn.ModuleDict({f"{i}_{j}": nn.Sequential(
            mlp(path, batchnorm=batchnorm, activate_last=True),
            mlp(path[::-1], batchnorm=batchnorm, softmax=True))
            for i in range(len(widths)) for j in range(len(widths)) if i != j})


class _PredictionClustering(DeepModel):
    supports_missing_views = True
    min_complete_samples = 2
    _dcp = False

    def __init__(self, n_clusters, *, embedding_dim=128, hidden_dims=(256, 256),
                 prediction_dims=(128, 256, 128), batchnorm=True, alpha=9.,
                 reconstruction_weight=.1, prediction_weight=.1, prediction_start=10,
                 max_iter=100, learning_rate=.0001, batch_size=256, n_init=10,
                 preprocess=True, random_state=None, device="auto", max_memory_mb=1024):
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.prediction_dims = prediction_dims
        self.batchnorm = batchnorm
        self.alpha = alpha
        self.reconstruction_weight = reconstruction_weight
        self.prediction_weight = prediction_weight
        self.prediction_start = prediction_start
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_init = n_init
        self.preprocess = preprocess
        self.random_state = random_state
        self.device = device
        self.max_memory_mb = max_memory_mb

    def fit(self, Xs, As=None, *, observed_mask=None):
        xs, mask = validate_views(Xs, observed_mask, missing=True, max_memory_mb=self.max_memory_mb)
        self._check_deep(len(xs[0]))
        integer("prediction_start", self.prediction_start, 0)
        number("alpha", self.alpha, positive=True)
        number("reconstruction_weight", self.reconstruction_weight, positive=True)
        number("prediction_weight", self.prediction_weight)
        if not self.prediction_dims:
            raise ValueError("prediction_dims must contain positive layer widths")
        for width in self.prediction_dims:
            integer("prediction_dims", width)
        if mask.all(1).sum() < 2:
            raise ValueError("Contrastive prediction requires at least two fully observed training samples")
        if not mask.all() and (self.prediction_start >= self.max_iter or self.prediction_weight <= 0):
            raise ValueError("Missing-view completion requires prediction_start < max_iter and prediction_weight > 0")
        paths = []
        for x in xs:
            dims = [x.shape[1], *self.hidden_dims, self.embedding_dim]
            paths.extend([dims, dims[::-1]])
        pred = [self.embedding_dim, *self.prediction_dims]
        paths.extend([pred, pred[::-1]] * (len(xs) * (len(xs) - 1)))
        parameter_budget(paths, self.max_memory_mb)
        scalers = fit_scalers(xs, mask, self.preprocess)
        prepared = apply_scalers(xs, mask, scalers)
        self._execute(lambda device: self._train(prepared, mask, device))
        self.scalers_ = scalers
        self.n_features_in_ = [x.shape[1] for x in xs]
        self.embedding_dim_ = self.embedding_.shape[1]
        self.n_complete_samples_ = int(mask.all(1).sum())
        return self

    def _train(self, xs, mask, device):
        with initialization_seed(self.random_state):
            model = PredictionNetwork([x.shape[1] for x in xs], self.hidden_dims, self.embedding_dim,
                                      self.prediction_dims, self.batchnorm)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        rng = np.random.default_rng(self.random_state)
        complete = np.flatnonzero(mask.all(1))
        history, components = [], []
        n_views = len(xs)
        pairs = [(i, j) for i in range(n_views) for j in range(i + 1, n_views)]
        for epoch in iterations(self.max_iter, "Prediction learning epochs"):
            model.train()
            values, terms = [], []
            for batch in batches(len(complete), self.batch_size, rng):
                idx = complete[batch]
                tensors = [torch.as_tensor(x[idx], device=device) for x in xs]
                outputs = [ae(x) for ae, x in zip(model.autoencoders, tensors)]
                zs = [item[0] for item in outputs]
                reconstruction = sum(F.mse_loss(out[1], x) for out, x in zip(outputs, tensors))
                contrastive = reconstruction.new_zeros(())
                prediction = reconstruction.new_zeros(())
                marginal = self.alpha if self._dcp else self.alpha + 1
                for i, j in pairs:
                    weight = .1 if self._dcp and n_views > 2 and (i, j) != (0, 1) else 1.
                    contrastive = contrastive + weight * consistency_loss(zs[i], zs[j], marginal)
                    if epoch >= self.prediction_start:
                        prediction = prediction + F.mse_loss(model.predictors[f"{i}_{j}"](zs[i]), zs[j])
                        prediction = prediction + F.mse_loss(model.predictors[f"{j}_{i}"](zs[j]), zs[i])
                if n_views > 2:
                    # DCP's complete-graph reference weighting; avoid its v-2 division at v=2.
                    reconstruction = reconstruction / (n_views - 2 if self._dcp else 1)
                    contrastive = contrastive / (n_views if self._dcp else len(pairs))
                    prediction = prediction / (n_views if self._dcp else len(pairs))
                loss = contrastive + self.reconstruction_weight * reconstruction + self.prediction_weight * prediction
                values.append(optimize(loss, optimizer))
                terms.append([float(reconstruction.detach()), float(contrastive.detach()), float(prediction.detach())])
            history.append(np.mean(values))
            components.append(np.mean(terms, axis=0))
        embedding = self._complete(model, xs, mask, device)
        km = KMeans(self.n_clusters, n_init=self.n_init, random_state=self.random_state).fit(embedding)
        return dict(model_=model.cpu().eval(), embedding_=embedding, labels_=km.labels_.astype(np.int32),
                    centroids_=km.cluster_centers_, inertia_=km.inertia_, kmeans_=km,
                    loss_history_=np.asarray(history), loss_components_=np.asarray(components),
                    loss_component_names_=["reconstruction", "consistency", "prediction"],
                    n_iter_=len(history), prediction_trained_=bool(self.prediction_start < self.max_iter
                                                                  and self.prediction_weight > 0))

    @torch.no_grad()
    def _complete(self, model, xs, mask, device):
        model.eval()
        result = []
        for idx in batches(len(xs[0]), self.batch_size):
            present = torch.as_tensor(mask[idx], device=device)
            zs = [torch.zeros((len(idx), self.embedding_dim), dtype=torch.float32, device=device) for _ in xs]
            for v, ae in enumerate(model.autoencoders):
                use = present[:, v]
                if use.any():
                    zs[v][use] = ae.encoder(torch.as_tensor(xs[v][idx][mask[idx, v]], device=device))
            # Only genuinely observed source views contribute; no cascaded predictions.
            completed = [z.clone() for z in zs]
            for v in range(len(xs)):
                missing = ~present[:, v]
                if not missing.any():
                    continue
                counts = torch.zeros(len(idx), device=device)
                for source in range(len(xs)):
                    use = missing & present[:, source]
                    if source != v and use.any():
                        completed[v][use] += model.predictors[f"{source}_{v}"](zs[source][use])
                        counts[use] += 1
                completed[v][missing] /= counts[missing, None].clamp_min(1)
            result.append(torch.cat(completed, dim=1).cpu().numpy())
        return np.concatenate(result)

    def transform(self, Xs, As=None, *, observed_mask=None):
        xs, mask = self._prediction_views(Xs, observed_mask)
        if not mask.all() and not self.prediction_trained_:
            raise ValueError("Missing-view prediction requires trained cross-view predictors")
        return self._complete(self.model_, xs, mask, torch.device("cpu"))

    def predict(self, Xs, As=None, *, observed_mask=None):
        embedding = self.transform(Xs, As, observed_mask=observed_mask)
        return self.kmeans_.predict(embedding)


@register_model("completer", description="Incomplete-view clustering via contrastive prediction and latent completion.",
                paper="https://openaccess.thecvf.com/content/CVPR2021/html/Lin_COMPLETER_Incomplete_Multi-View_Clustering_via_Contrastive_Prediction_CVPR_2021_paper.html",
                implementation="https://github.com/XLearning-SCU/2021-CVPR-Completer")
class COMPLETER(_PredictionClustering):
    """Two-view COMPLETER, with an all-pairs extension for additional views.

    Training uses fully observed rows, as in the reference. Missing latent
    views are predicted from observed ones. With one view the objective reduces
    to autoencoder reconstruction plus K-means, not contrastive learning.
    ``max_iter`` counts epochs; ``prediction_start`` is a zero-based epoch.
    Compact default hidden widths can be replaced with reference widths.
    """


Completer = COMPLETER
