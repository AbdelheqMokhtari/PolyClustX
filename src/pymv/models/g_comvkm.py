"""Globally collaborative K-means with entropy-weighted feature reduction."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from pymv.progress import iterations

from ._common import TorchModel, apply_scalers, fit_scalers, memory_check, number, validate_views
from .registry import register_model


def weighted_distances(x, centers, weights):
    return ((x.square() * weights).sum(1, keepdim=True)
            + (centers.square() * weights).sum(1)[None, :] - 2 * (x * weights) @ centers.T).clamp_min(0)


@register_model("g_comvkm", description="Globally collaborative K-means with feature selection and view weighting.",
                paper="https://doi.org/10.3390/electronics14112129",
                implementation="https://github.com/KristinaP09/G-CoMVKM")
class GCoMVKM(TorchModel):
    """PyTorch adaptation of collaborative membership/center/feature/view updates.

    Uses the original demonstration's gamma=5, theta=4. ``gamma > 1`` avoids
    the singular view-weight update and rewards lower distortion. Feature
    regularizers remain vectors, stable softmax prevents underflow, and every
    view keeps at least its highest-weight feature. Original feature indices
    are retained for prediction; user arrays are never reduced in place.
    """

    def __init__(self, n_clusters, *, gamma=5., theta=4., max_iter=100, tol=.0001,
                 feature_selection=True, preprocess=True, random_state=None, device="auto", max_memory_mb=1024):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.theta = theta
        self.max_iter = max_iter
        self.tol = tol
        self.feature_selection = feature_selection
        self.preprocess = preprocess
        self.random_state = random_state
        self.device = device
        self.max_memory_mb = max_memory_mb

    def fit(self, Xs, As=None, *, observed_mask=None):
        xs, mask = validate_views(Xs, observed_mask, dtype=np.float64, max_memory_mb=self.max_memory_mb)
        self._check_common(len(xs[0]))
        number("gamma", self.gamma, minimum=1, positive=True)
        number("theta", self.theta, positive=True)
        number("tol", self.tol)
        memory_check(len(xs[0]) * self.n_clusters * len(xs) * 8 * 12, self.max_memory_mb)
        scalers = fit_scalers(xs, mask, self.preprocess)
        self._execute(lambda device: self._solve(apply_scalers(xs, mask, scalers), device))
        self.scalers_, self.n_features_in_ = scalers, [x.shape[1] for x in xs]
        self.n_samples_fit_ = len(xs[0])
        self.embedding_dim_ = self.embedding_.shape[1]
        return self

    def _memberships(self, xs, centers, feature_weights, delta, view_weights, n_train):
        distances = [weighted_distances(x, c, w * d) for x, c, w, d in zip(xs, centers, feature_weights, delta)]
        scaled = [v * d for v, d in zip(view_weights, distances)]
        global_distance = (1 - self.gamma * n_train) * sum(scaled)
        memberships = [F.one_hot((self.gamma * n_train * d - global_distance).argmin(1), self.n_clusters)
                       .to(x.dtype) for d, x in zip(scaled, xs)]
        return memberships, distances

    @torch.no_grad()
    def _solve(self, arrays, device):
        xs = [torch.as_tensor(x, dtype=torch.float64, device=device) for x in arrays]
        n, views = len(xs[0]), len(xs)
        ids = np.random.default_rng(self.random_state).choice(n, self.n_clusters, replace=False)
        centers = [x[ids].clone() for x in xs]
        selected = [torch.arange(x.shape[1], device=device) for x in xs]
        features = [torch.full((x.shape[1],), 1 / x.shape[1], dtype=x.dtype, device=device) for x in xs]
        spans = [x.max(0).values - x.min(0).values for x in xs]
        delta = [1 / (n * torch.where(span > 0, span, 1)) for span in spans]
        weights = torch.full((views,), 1 / views, dtype=torch.float64, device=device)
        history = []
        for _ in iterations(self.max_iter, "Collaborative feature/view updates"):
            membership, _ = self._memberships(xs, centers, features, delta, weights, n)
            total_membership = sum(membership)
            mixtures = [(1 - self.gamma) * u + self.gamma * total_membership for u in membership]
            threshold = 1 / sum(x.shape[1] for x in xs)
            for v in range(views):
                x, u = xs[v], mixtures[v]
                mass = u.sum(0)
                means = u.T @ x / mass.clamp_min(1e-12)[:, None]
                centers[v] = torch.where((mass > 0)[:, None], means, centers[v])
                # Sum weighted squared residuals per feature without allocating N×K×D.
                distortion = (u.sum(1)[:, None] * x.square()).sum(0) \
                    - 2 * ((u.T @ x) * centers[v]).sum(0) + (mass[:, None] * centers[v].square()).sum(0)
                cost = distortion.clamp_min(0) * ((x.shape[1] + n) / n) * weights[v].pow(self.theta) * delta[v]
                features[v] = torch.softmax(-cost / self.theta - delta[v].log(), dim=0)
                keep = features[v] >= threshold if self.feature_selection else torch.ones_like(features[v], dtype=bool)
                keep[features[v].argmax()] = True
                selected[v] = selected[v][keep]
                xs[v], centers[v], delta[v] = x[:, keep], centers[v][:, keep], delta[v][keep]
                features[v] = features[v][keep] / features[v][keep].sum()
            distances = [weighted_distances(x, c, w * d) for x, c, w, d in zip(xs, centers, features, delta)]
            global_distance = self.gamma * self.theta * sum(distances)
            energies = torch.stack([(u * d).sum() + ((total_membership - own) * global_distance).sum()
                                    for u, d, own in zip(mixtures, distances, membership)])
            weights = torch.softmax(-energies.clamp_min(1e-30).log() / (self.gamma - 1), dim=0)
            local = sum((u * d).sum() * w.pow(self.gamma) / x.shape[1]
                        for u, d, w, x in zip(membership, distances, weights, xs))
            collaborative = sum(weights[v].square()
                                * (mixtures[v] * distances[v] - mixtures[j] * distances[j]).abs().sum()
                                for v in range(views) for j in range(views))
            entropy = self.theta * sum((w * (d * w).clamp_min(1e-30).log()).sum().abs()
                                       for w, d in zip(features, delta))
            loss = float(local + collaborative + entropy)
            if not np.isfinite(loss):
                raise ValueError("G-CoMVKM objective overflowed; rescale features or increase theta")
            history.append(loss)
            if len(history) > 1 and abs(history[-1] - history[-2]) <= self.tol:
                break
        membership, distances = self._memberships(xs, centers, features, delta, weights, n)
        labels = sum(w * u for w, u in zip(weights, membership)).argmax(1)
        embedding = torch.cat([x * (w * f * d).sqrt() for x, w, f, d in zip(xs, weights, features, delta)], dim=1)
        return dict(labels_=labels.cpu().numpy(), embedding_=embedding.cpu().numpy(),
                    centroids_=[c.cpu().numpy() for c in centers], view_weights_=weights.cpu().numpy(),
                    feature_weights_=[w.cpu().numpy() for w in features], delta_=[d.cpu().numpy() for d in delta],
                    selected_features_=[i.cpu().numpy() for i in selected],
                    per_view_results_=[u.argmax(1).cpu().numpy() for u in membership],
                    loss_history_=np.asarray(history), n_iter_=len(history),
                    inertia_=float(sum(w * d for w, d in zip(weights, distances)).min(1).values.sum()))

    def transform(self, Xs):
        xs, _ = self._prediction_views(Xs)
        entries = zip(xs, self.selected_features_, self.view_weights_, self.feature_weights_, self.delta_)
        return np.concatenate([x[:, i] * np.sqrt(v * w * d) for x, i, v, w, d in entries], 1)

    def predict(self, Xs):
        xs, _ = self._prediction_views(Xs)
        # The training sample count, not prediction batch size, defines collaboration strength.
        tensors = [torch.as_tensor(x[:, i], dtype=torch.float64) for x, i in zip(xs, self.selected_features_)]
        centers, features, delta = [[torch.as_tensor(x) for x in values]
                                    for values in (self.centroids_, self.feature_weights_, self.delta_)]
        weights = torch.as_tensor(self.view_weights_)
        membership, _ = self._memberships(tensors, centers, features, delta, weights, self.n_samples_fit_)
        return sum(w * u for w, u in zip(weights, membership)).argmax(1).numpy()
