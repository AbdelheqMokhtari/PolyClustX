# The Eight LMGEC Extensions — Formulations & Objective Changes

This document specifies every modification introduced on top of the
original LMGEC algorithm, one by one: what changes in the **objective
function**, how the **update rules** change, and how to enable each
extension **alone** in `pymv` (which is exactly what the default
ablation ladder in `build_ablation_suite` does).

Companion documents:
* `docs/LMGEC_modular_architecture.md` — the software architecture.
* `notebooks/lmgec_ablation_study.ipynb` — runnable ablation study.

---

## 0. The baseline: original LMGEC (all extensions off)

Given views $X^{(v)} \in \mathbb{R}^{n \times d_v}$ with adjacencies
$A^{(v)}$, LMGEC first builds propagated features

$$H^{(v)} = S^{(v)} X^{(v)}, \qquad S^{(v)} = D^{-1}\!\big(A^{(v)} + \beta I\big),$$

(after row L2 / TF-IDF normalization) and solves

$$\min_{\substack{W^{(v)\top} W^{(v)} = I \\ F,\; G}}
\; \sum_{v=1}^{V} \alpha_v \big\| H^{(v)} - G F\, W^{(v)\top} \big\|_F^2$$

by alternating: orthogonal-Procrustes update of each projection
$W^{(v)}$, nearest-centroid assignment $G$, and segment-mean centroids
$F$, on the consensus embedding $Z = \sum_v \alpha_v H^{(v)} W^{(v)}$.
The view weights are fixed **a priori**:

$$\alpha_v \propto \exp\!\Big(-\tfrac{\mathrm{inertia}_v}{\sqrt{n}\,\tau}\Big).$$

**Correctness of the baseline.** `LMGECplus` with every extension off
reproduces this exactly: it was verified **bit-for-bit** against the
pre-refactor implementation on 12 configurations (fixed
`random_state`), and statistically against the original TF `LMGEC`
class in `tests/test_lmgecplus.py::TestBackwardCompat`.

```python
baseline = LMGECplus(n_clusters=k, embedding_dim=m)   # = original LMGEC
```

---

## E1 — Robust reconstruction via the ℓ2,1 norm

**Motivation.** The squared Frobenius norm is dominated by outlier
*nodes* (corrupted samples, misaligned modalities).

**Objective change.** Replace, per view, the squared norm by the ℓ2,1
norm (row-wise ℓ2, column-wise ℓ1):

$$\big\| H^{(v)} - G F W^{(v)\top} \big\|_F^2
\;\longrightarrow\;
\big\| H^{(v)} - G F W^{(v)\top} \big\|_{2,1}
= \sum_{i=1}^{n} \big\| (H^{(v)} - G F W^{(v)\top})_{i\cdot} \big\|_2 .$$

**Essential changes to the updates (IRLS).** Minimising the ℓ2,1 norm
is equivalent to iteratively re-weighted least squares with per-node
weights

$$u_i^{(v)} = \frac{1}{2\, r_i^{(v)}}, \qquad
r_i^{(v)} = \big\|(H^{(v)} - T\,W^{(v)\top})_{i\cdot}\big\|_2 ,$$

(rescaled to mean 1 to preserve the problem scale) so that **both**
coupled updates become weighted:

* **W-update** → *weighted* orthogonal Procrustes:
  $W^{(v)} = U V^\top$ from
  $\mathrm{svd}\big(H^{(v)\top}\,\mathrm{diag}(\beta^{(v)} u^{(v)})\,T\big)$;
* **centroid update** → *weighted* per-cluster mean with node weights
  $\sum_v \beta^{(v)}_i u^{(v)}_i$;
* the **monitored loss** uses $\rho(r) = r$ instead of $r^2$.

Nodes with large residuals are automatically down-weighted, which is
the source of the robustness.

```python
LMGECplus(..., reconstruction_loss="l21")     # legacy: robust_loss=True
```

Implementation: `losses.L21Loss` (a Huber loss `"huber"` is available
through the same IRLS contract).

---

## E2 — Node-level view attention

**Motivation.** A single global $\alpha_v$ assumes a view is equally
reliable for *every* node. In heterogeneous data, view quality varies
per sample.

**Objective change.** Replace the global weights by per-node fusion
weights $\beta_{v,i}$ with $\sum_v \beta_{v,i} = 1$ for each node $i$,
chosen to minimise the attended reconstruction error:

$$\min_{\beta \ge 0,\; \sum_v \beta_{v,i}=1}\;
\sum_{v}\sum_{i} \beta_{v,i}\, e_{v,i}
\quad\text{with}\quad
e_{v,i} = \big\|(H^{(v)} - T W^{(v)\top})_{i\cdot}\big\|_2^2 ,$$

whose (entropy-smoothed, see E5) closed form is the softmax

$$\beta_{v,i} = \frac{\exp(-e_{v,i}/\lambda)}{\sum_{w}\exp(-e_{w,i}/\lambda)} .$$

The consensus becomes a **per-node** convex fusion
$Z_{i\cdot} = \sum_v \beta_{v,i}\, (H^{(v)} W^{(v)})_{i\cdot}$, and the
same $\beta$'s weight the W-updates. In this variant the attention is
computed **once** after initialisation (its dynamic recomputation is
extension E4, ablated separately).

```python
LMGECplus(..., attention="static")            # legacy: node_attention=True,
                                              #         dynamic_attention=False
```

Implementation: `attention.NodeLevelAttention` /
`attention.StaticAttention`, consumed through
`weighting.AttentionWeighting`.

---

## E3 — Dimension-normalized reconstruction errors

**Motivation.** Raw errors $e_{v,i}$ scale with the view
dimensionality $d_v$: a 2048-d CNN view produces mechanically larger
residuals than a 300-d text view, biasing any error-driven weighting
toward low-dimensional views.

**Change.** Normalize the per-node error by the view dimension before
the attention allocation:

$$\tilde{e}_{v,i} = \frac{e_{v,i}}{d_v} .$$

This only alters the *input* of the attention softmax — the objective
and updates of E2 are otherwise unchanged, which is why the ablation
variant is *attention + dim-norm* (the normalisation has nothing to
act on without attention).

```python
LMGECplus(..., attention={"name": "static", "dim_normalized": True})
```

Implementation: `NodeLevelAttention(dim_normalized=True)`.

---

## E4 — Dynamic view weights (α recomputed every iteration)

**Motivation.** The baseline fixes $\alpha_v$ *before* optimisation
from the initial K-Means inertia; but a view's usefulness changes as
the shared representation improves.

**Change.** Re-solve the weight allocation at **every iteration** $t$
from the current reconstruction errors:

$$\alpha_v^{(t)} \propto
\exp\!\Big(-\bar{e}_v^{(t)}/\lambda\Big),
\qquad
\bar{e}_v^{(t)} = \tfrac{1}{n}\sum_i e_{v,i}^{(t)} ,$$

i.e. one adaptive scalar per view per iteration (view-level dynamic
attention). Combined with E2 this becomes dynamic *node-level*
attention (`attention="dynamic"`).

```python
LMGECplus(..., attention="view")              # dynamic scalar alphas
LMGECplus(..., attention="dynamic")           # + node level (E2∧E4)
```

Implementation: `attention.ViewLevelAttention(dynamic=True)`; the
per-iteration α trajectory is observable via
`model.results_.view_weights` / `attention_weights`.

---

## E5 — Learned weight regularisation (entropy or ℓ2)

**Motivation.** The bare allocation problem of E2/E4 is a linear
program whose solution is degenerate (all mass on the best view). A
regulariser makes the weights *learnable quantities* with a controlled
sharpness, re-solved in closed form at every step.

**Objective change.** Add a regulariser $\Omega(\beta)$ to the
allocation problem:

$$\min_{\beta \ge 0,\;\sum_v \beta_{v,i}=1}\;
\sum_{v} \beta_{v,i}\, \tilde e_{v,i} \;+\; \lambda\, \Omega(\beta_{\cdot,i}) .$$

* **Entropy** $\Omega = \sum_v \beta_v \log \beta_v$ → closed form is
  the **softmax with temperature λ** (smooth, dense weights):
  $\beta_{v,i} \propto \exp(-\tilde e_{v,i}/\lambda)$.
* **ℓ2** $\Omega = \|\beta\|_2^2$ → closed form is the **Euclidean
  projection of $-\tilde e/(2\lambda)$ onto the probability simplex**
  (can produce exact zeros → sparse view selection).

Errors are median-rescaled first so λ is a unitless, dataset-independent
temperature.

```python
LMGECplus(..., attention={"name": "node", "regularization": "entropy"})
LMGECplus(..., attention={"name": "node", "regularization": "l2"})
# λ: attention={"name": "node", "regularization": "entropy", "lam": 0.5}
```

Implementation: `components.view_attention` (softmax / simplex
projection `components.simplex_project`).

---

## E6 — Controlled graph propagation

**Motivation.** LMGEC's binary choice — full propagation $H = SX$ or
none — either over-smooths (dense graphs) or discards the graph
entirely.

**Proposed formulation.** A continuous trade-off with a view-specific
smoothing parameter $\gamma_v \in [0, 1]$:

$$H^{(v)} = \big((1-\gamma_v)\, I + \gamma_v\, S^{(v)}\big)\, X^{(v)} .$$

$\gamma_v = 0$ → raw (normalized) features; $\gamma_v = 1$ → original
LMGEC; intermediate values blend identity and smoothing. With
`gamma="auto"`, $\gamma_v$ is set per view from the graph-signal
smoothness energy
$\mathrm{tr}(H^\top L H)/\|H\|_F^2$, $L = I - S$ (rough signals get
more propagation).

This is a **preprocessing** change: the optimisation objective is
untouched, only the $H^{(v)}$ it operates on changes.

```python
LMGECplus(..., graph_propagation={"enabled": True, "gamma": 0.5})
LMGECplus(..., graph_propagation={"enabled": True, "gamma": "auto"})
# per view: graph_propagation=[{"enabled": True, "gamma": 0.3},
#                              {"enabled": True, "gamma": 0.8}]
```

Implementation: `propagation.ControlledPropagation`
(`components.controlled_propagation`, `graph_smoothness`).

---

## E7 — Soft clustering assignments

**Motivation.** Hard K-Means assignments make brittle all-or-nothing
decisions early in training and give no confidence information.

**Objective change.** Replace the binary indicator $G \in \{0,1\}^{n
\times k}$ by a row-stochastic responsibility matrix $R \in [0,1]^{n
\times k}$, $\sum_j R_{ij} = 1$:

$$\min_{W, F, R}\;
\sum_v \alpha_v \big\| H^{(v)} - \mathbf{R}\, F\, W^{(v)\top} \big\|^2 ,
\qquad
R_{ij} = \frac{\exp(-\beta_s \|z_i - f_j\|^2)}
              {\sum_{l}\exp(-\beta_s \|z_i - f_l\|^2)} .$$

**Changes to the updates.**

* the reconstruction **target** becomes the responsibility-weighted
  centroid mix $T = R F$ (instead of $F[G]$) — this is the essential
  objective modification;
* the **centroid update** becomes the responsibility-weighted mean
  $F = (R^\top Z) \oslash (R^\top \mathbf{1})$;
* $\beta_s$ controls sharpness ($\beta_s \to \infty$ recovers hard
  K-Means); hard labels for evaluation are $\arg\max_j R_{ij}$.

```python
LMGECplus(..., clustering="soft")                       # β_s = 50
LMGECplus(..., clustering={"method": "soft", "beta": 20.0})
LMGECplus(..., clustering="gmm")                        # GMM head variant
```

Implementation: `clustering.SoftKMeansHead` (probabilities exposed as
`results_.cluster_probabilities`). A `GMMHead` offers full Gaussian
responsibilities through the same interface.

---

## E8 — Cross-view representation alignment (proposed formulation)

**Motivation.** Each view's projection $Z^{(v)} = H^{(v)} W^{(v)}$ is
pulled only toward the *clustering* target; nothing prevents the
per-view embeddings from drifting apart in the shared space
$\mathbb{R}^p$, which degrades the consensus.

**Proposal.** Assuming a shared embedding dimension $p$ across views,
add a lightweight alignment penalty that pulls every view's embedding
toward the consensus of the previous iterate $\bar Z$:

$$\mathcal{L}_{\text{align}}
= \mu \sum_{v=1}^{V} \big\| H^{(v)} W^{(v)} - \bar{Z} \big\|_F^2 ,
\qquad \mu > 0 \text{ small (e.g. } 0.1\text{)} .$$

**Why this form.** Expanding the per-view W-subproblem,

$$\big\|H^{(v)}W^{(v)} - T\big\|_F^2 + \mu \big\|H^{(v)}W^{(v)} - \bar Z\big\|_F^2$$

is (up to constants, using $W^\top W = I$) minimised by the **same
orthogonal Procrustes update with an augmented target**

$$T^{(v)} = T + \mu\, \bar{Z} ,$$

so the alignment costs one matrix addition and keeps every update
closed-form — no gradient steps, no extra parameters beyond μ. It acts
as a temporal/cross-view consistency term: views agree with the fused
representation, and the fused representation changes smoothly.

Two alternative alignment geometries are provided through the same
interface, for comparison:

* **cosine** — align *directions* only (row-normalized $\bar Z$ in the
  augmentation): magnitude-dominant nodes stop dominating the pull;
* **contrastive** — pull toward the own consensus row while pushing
  away from the batch mean ($T + \mu(\bar Z - \overline{\bar Z})$), a
  linear one-negative surrogate of a contrastive loss (a full InfoNCE
  has no closed-form W-update).

```python
LMGECplus(..., alignment="consensus")     # proposed penalty (μ=0.1)
LMGECplus(..., cross_view_alignment=True, align_mu=0.2)   # legacy flags
LMGECplus(..., alignment="cosine")        # direction-only variant
LMGECplus(..., alignment="contrastive")   # pull/push variant
```

Implementation: `alignment.ConsensusAlignment` (plus
`CosineAlignment`, `ContrastiveAlignment`); custom penalties subclass
`Alignment.augment_target`.

---

## Running the ablation

Each extension is one override; the standard ladder is prebuilt:

```python
from pymv.experiment import Experiment
from pymv.metrics import ari_score, clustering_accuracy, nmi_score
from pymv.models.lmgecplus import build_ablation_suite

suite = build_ablation_suite(n_clusters=k, embedding_dim=16,
                             max_iter=30, random_state=0)
# baseline, E1 robust-l21, E2 node-attention, E3 attn+dim-norm,
# E4 dynamic-alphas, E5 attn+entropy-reg, E6 controlled-prop,
# E7 soft-clustering, E8 alignment, full

Experiment(datasets=["acm"], models=list(suite.values()),
           metrics=[clustering_accuracy, nmi_score, ari_score],
           runs=5, save_markdown=True).run()
```

The runnable version (with plots, sensitivity analysis and the result
object walkthrough) is `notebooks/lmgec_ablation_study.ipynb`.

Quick sanity tests (light, synthetic data only):

```bash
python -m pytest tests/test_lmgec_components.py tests/test_lmgecplus.py -q
```
