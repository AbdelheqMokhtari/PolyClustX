"""Multi-level feature learning: reconstruction, feature/label contrast and self-training."""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from polyclustx.progress import iterations

from ._common import apply_scalers, batches, fit_scalers, integer, number, optimize, validate_views
from ._deep import Autoencoder, DeepModel, linear, parameter_budget
from ._torch_utils import initialization_seed
from .registry import register_model
from .scm.scm import SCM


def label_contrastive_loss(left, right, temperature):
    """Cluster-column NT-Xent plus marginal negative entropy."""
    entropy = left.new_zeros(())
    for q in (left, right):
        marginal = q.mean(0)
        entropy = entropy + np.log(q.shape[1]) + (marginal * marginal.clamp_min(1e-12).log()).sum()
    return SCM._contrastive_loss(F.normalize(left.T, dim=1), F.normalize(right.T, dim=1), temperature) + entropy


class MultiLevelNetwork(nn.Module):
    def __init__(self, widths, hidden, latent, projection, clusters):
        super().__init__()
        self.autoencoders = nn.ModuleList([Autoencoder(w, hidden, latent) for w in widths])
        self.projector = linear(latent, projection)
        self.cluster_head = linear(latent, clusters)

    def forward(self, xs):
        zs, reconstructed = zip(*(ae(x) for ae, x in zip(self.autoencoders, xs)))
        projected = [F.normalize(self.projector(z), dim=1) for z in zs]
        logits = [self.cluster_head(z) for z in zs]
        return projected, logits, reconstructed


def aligned_pseudo_labels(pseudo, predicted, k):
    table = np.zeros((k, k), dtype=np.int64)
    np.add.at(table, (pseudo, predicted), 1)
    rows, columns = linear_sum_assignment(-table)
    mapping = np.empty(k, dtype=np.int64)
    mapping[rows] = columns
    return mapping[pseudo]


@register_model("mflvc", description="Multi-level contrastive features and cluster assignments with self-training.",
                paper="https://openaccess.thecvf.com/content/CVPR2022/html/Xu_Multi-Level_Feature_Learning_for_Contrastive_Multi-View_Clustering_CVPR_2022_paper.html",
                implementation="https://github.com/SubmissionsIn/MFLVC")
class MFLVC(DeepModel):
    """Three-stage MFLVC with shared feature and cluster heads.

    ``pretrain_epochs`` reconstruction epochs precede ``max_iter`` contrastive
    epochs and ``finetune_epochs`` pseudo-label epochs. Final assignments use
    the mean cluster probabilities, not external labels. ``embedding_`` is the
    concatenation of high-level L2-normalized view features. With a single view,
    cross-view terms vanish; reconstruction and pseudo-label training remain.
    """

    def __init__(self, n_clusters, *, embedding_dim=512, projection_dim=128,
                 hidden_dims=(500, 500, 2000), pretrain_epochs=20, max_iter=50, finetune_epochs=10,
                 temperature=.5, label_temperature=1., learning_rate=.0003, weight_decay=0.,
                 batch_size=256, n_init=10, preprocess=True, random_state=None,
                 device="auto", max_memory_mb=1024):
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.hidden_dims = hidden_dims
        self.pretrain_epochs = pretrain_epochs
        self.max_iter = max_iter
        self.finetune_epochs = finetune_epochs
        self.temperature = temperature
        self.label_temperature = label_temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_init = n_init
        self.preprocess = preprocess
        self.random_state = random_state
        self.device = device
        self.max_memory_mb = max_memory_mb

    def fit(self, Xs, As=None, *, observed_mask=None):
        xs, mask = validate_views(Xs, observed_mask, max_memory_mb=self.max_memory_mb)
        self._check_deep(len(xs[0]))
        integer("projection_dim", self.projection_dim)
        integer("pretrain_epochs", self.pretrain_epochs)
        integer("finetune_epochs", self.finetune_epochs)
        number("temperature", self.temperature, positive=True)
        number("label_temperature", self.label_temperature, positive=True)
        number("weight_decay", self.weight_decay)
        paths = [[self.embedding_dim, self.projection_dim], [self.embedding_dim, self.n_clusters]]
        for x in xs:
            path = [x.shape[1], *self.hidden_dims, self.embedding_dim]
            paths.extend([path, path[::-1]])
        parameter_budget(paths, self.max_memory_mb)
        scalers = fit_scalers(xs, mask, self.preprocess)
        self._execute(lambda device: self._train(apply_scalers(xs, mask, scalers), device))
        self.scalers_ = scalers
        self.n_features_in_ = [x.shape[1] for x in xs]
        self.embedding_dim_ = self.embedding_.shape[1]
        return self

    @torch.no_grad()
    def _infer(self, model, xs, device):
        model.eval()
        hs, qs = [[] for _ in xs], [[] for _ in xs]
        for idx in batches(len(xs[0]), self.batch_size):
            high, logits, _ = model([torch.as_tensor(x[idx], device=device) for x in xs])
            for v in range(len(xs)):
                hs[v].append(high[v].cpu().numpy())
                qs[v].append(logits[v].softmax(1).cpu().numpy())
        return [np.concatenate(h) for h in hs], [np.concatenate(q) for q in qs]

    def _train(self, xs, device):
        with initialization_seed(self.random_state):
            model = MultiLevelNetwork([x.shape[1] for x in xs], self.hidden_dims, self.embedding_dim,
                                      self.projection_dim, self.n_clusters)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        rng = np.random.default_rng(self.random_state)
        history, stages, components = [], [], []
        for stage, epochs in (("pretrain", self.pretrain_epochs), ("contrastive", self.max_iter),
                              ("finetune", self.finetune_epochs)):
            if stage == "finetune":
                hs, _ = self._infer(model, xs, device)
                pseudo = []
                for h in hs:
                    span = np.ptp(h, axis=0)
                    scaled = (h - h.min(0)) / np.where(span > 0, span, 1)
                    pseudo.append(KMeans(self.n_clusters, n_init=self.n_init,
                                         random_state=self.random_state).fit_predict(scaled))
            for _ in iterations(epochs, stage.title() + " epochs"):
                if stage == "finetune":
                    _, qs = self._infer(model, xs, device)
                    targets = [aligned_pseudo_labels(p, q.argmax(1), self.n_clusters) for p, q in zip(pseudo, qs)]
                model.train()
                values, terms = [], []
                for idx in batches(len(xs[0]), self.batch_size, rng):
                    tensors = [torch.as_tensor(x[idx], device=device) for x in xs]
                    hs, logits, decoded = model(tensors)
                    reconstruction = sum(F.mse_loss(x, xr) for x, xr in zip(tensors, decoded))
                    feature = reconstruction.new_zeros(())
                    label = reconstruction.new_zeros(())
                    if stage == "contrastive":
                        qs = [logit.softmax(1) for logit in logits]
                        for i in range(len(xs)):
                            for j in range(i + 1, len(xs)):
                                feature = feature + SCM._contrastive_loss(hs[i], hs[j], self.temperature)
                                label = label + label_contrastive_loss(qs[i], qs[j], self.label_temperature)
                    if stage == "finetune":
                        # Use logits, not probabilities passed through another softmax.
                        loss = sum(F.cross_entropy(logit, torch.as_tensor(t[idx], device=device))
                                   for logit, t in zip(logits, targets))
                    else:
                        loss = reconstruction + feature + label
                    values.append(optimize(loss, optimizer))
                    terms.append([float(reconstruction.detach()), float(feature.detach()), float(label.detach()),
                                  float(loss.detach()) if stage == "finetune" else 0.])
                history.append(np.mean(values))
                stages.append(stage)
                components.append(np.mean(terms, axis=0))
        hs, qs = self._infer(model, xs, device)
        embedding = np.concatenate(hs, axis=1)
        probabilities = np.mean(qs, axis=0)
        labels = probabilities.argmax(1)
        centers = np.zeros((self.n_clusters, embedding.shape[1]), dtype=np.float32)
        for k in range(self.n_clusters):
            if np.any(labels == k):
                centers[k] = embedding[labels == k].mean(0)
        return dict(model_=model.cpu().eval(), labels_=labels, embedding_=embedding, centroids_=centers,
                    probabilities_=probabilities, view_labels_=[q.argmax(1) for q in qs],
                    loss_history_=np.asarray(history), loss_stages_=stages, loss_components_=np.asarray(components),
                    loss_component_names_=["reconstruction", "feature_contrast", "label_contrast",
                                           "pseudo_label_cross_entropy"],
                    n_iter_=len(history), inertia_=float(np.square(embedding - centers[labels]).sum()))

    def transform(self, Xs, As=None):
        xs, _ = self._prediction_views(Xs)
        hs, _ = self._infer(self.model_, xs, torch.device("cpu"))
        return np.concatenate(hs, axis=1)

    def predict(self, Xs, As=None):
        xs, _ = self._prediction_views(Xs)
        _, qs = self._infer(self.model_, xs, torch.device("cpu"))
        return np.mean(qs, axis=0).argmax(1)
