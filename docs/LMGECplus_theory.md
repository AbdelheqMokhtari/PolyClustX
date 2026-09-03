# LMGECplus — Theoretical Report

> Modular heterogeneous extension of LMGEC.
> Implementation: [`src/pymv/models/lmgecplus/`](../src/pymv/models/lmgecplus/)
> (`lmgecplus.py` = model, `components.py` = the extension primitives).
> Companion experimental report: [`LMGECplus_experiments.md`](LMGECplus_experiments.md).

A note on the repository layout. The task brief sketches a
`library/algorithms/lmgec(plus)/` tree; the actual project is the
`pymv` package, so LMGECplus lives at `src/pymv/models/lmgecplus/`
next to `lmgec/`, which keeps it integrated with the existing
registry, CLI, experiment runner, and random search.

---

## 1. Motivation

LMGEC fuses several views into one shared latent space with a fast,
closed-form alternating scheme (orthogonal projections + global view
weights + hard K-Means). It is excellent on clean, homogeneous graph
data but makes three assumptions that break in **heterogeneous /
multimodal** settings (e.g. CLIP image + text, or views of very
different dimensionality):

1. **One scalar weight per view** (`α_v`) — a whole modality is
   trusted uniformly across all samples, even though corruption and
   informativeness are usually *node-specific*.
2. **Squared (ℓ₂) reconstruction loss** — quadratically sensitive to
   outlier / corrupted nodes and modality misalignment.
3. **Full graph propagation** `H = SX` and **hard assignments** — can
   over-smooth and force brittle 0/1 decisions on boundary samples.

LMGECplus introduces eight *independently toggleable* extensions that
relax these assumptions. Crucially, with **all** extensions off it
reduces **exactly** to LMGEC (verified empirically, parity diff 0.000,
and in `test_backward_compat`).

---

## 2. Notation

| Symbol | Meaning |
|--------|---------|
| `V`, `n`, `k`, `m` | #views, #samples, #clusters, latent dim |
| `H_v ∈ ℝ^{n×d_v}` | preprocessed features of view `v` |
| `W_v ∈ ℝ^{d_v×m}` | orthogonal projection (`W_vᵀW_v = I`) |
| `Z_v = H_v W_v`, `Z` | per-view / consensus embedding |
| `F ∈ ℝ^{k×m}`, `G` | centroids, assignment (hard labels or soft `R∈ℝ^{n×k}`) |
| `β_{v,i}` | node-level fusion weight (`Σ_v β_{v,i}=1`) |
| `u_{v,i}` | robust (IRLS) node weight |
| `S_v` | row-normalised propagation matrix `D⁻¹(A_v+βI)` |

---

## 3. Base objective (LMGEC) and the LMGECplus generalisation

LMGEC minimises, in alternating closed form,

```
  min_{ {W_v}, F, G }   Σ_v α_v ‖ H_v − F[G] W_vᵀ ‖_F²,   W_vᵀW_v = I.   (LMGEC)
```

LMGECplus optimises the more general objective

```
  min  Σ_v Σ_i β_{v,i} · ρ( ‖ (H_v − T W_vᵀ)_i ‖₂ )  +  μ Σ_{v<w} ‖Z_v − Z_w‖_F²   (★)
```

where

* `T` is the reconstruction target — `F[G]` (hard) or `R F` (soft);
* `ρ(x)=x²` (standard) or `ρ(x)=x` (**robust ℓ2,1**, extension 1);
* `β_{v,i}` is a **node-level** weight (extensions 2–5);
* the last term is **cross-view alignment** (extension 8, weight `μ`);
* the features `H_v` themselves are produced by **controlled
  propagation** (extension 6);
* assignments may be **soft** (extension 7).

Setting `ρ=(·)²`, `β_{v,i}=α_v`, `μ=0`, hard `G`, and `H_v=S_vX_v`
recovers (LMGEC) term-for-term.

The four block updates remain closed-form:

* **Projection** (weighted orthogonal Procrustes). For
  `min_W Σ_i w_i ‖H_{v,i} − T_i Wᵀ‖²` s.t. `WᵀW=I`, with
  `M = H_vᵀ diag(w) T = UΣVᵀ`, the optimum is `W_v = U Vᵀ`.
* **Consensus** `Z = Σ_v β_{v}·Z_v` (per-node convex combination).
* **Assignment** `G_i = argmin_c‖Z_i − f_c‖²` (hard) or
  `R = softmax(−β_s ‖Z_i−f_c‖²)` (soft).
* **Centroids** `F = (Qᵀ diag(w) Z) ⊘ (Qᵀ diag(w) 1)` with `Q` the
  (one-hot or soft) assignment — i.e. a (weighted) responsibility mean.

Because each subproblem is solved with the other blocks fixed and the
weights `β,u` use the previous iterate (as LMGEC does for `G,F`), the
scheme is a block-coordinate / IRLS hybrid that monotonically improves
each block's local objective.

---

## 4. The eight extensions

### 4.1 Robust ℓ2,1 reconstruction loss (ext. 1)

Replacing `Σ_i‖r_i‖²` by `Σ_i‖r_i‖` (ℓ2,1) makes the per-view loss grow
**linearly** rather than quadratically with a node's residual, bounding
the influence of corrupted / misaligned nodes. The ℓ2,1 minimiser is
obtained by IRLS: each outer iteration reweights node `i` by

```
  u_{v,i} = 1 / ( 2 ‖(H_v − T W_vᵀ)_i‖₂ + ε ),
```

and solves the weighted Procrustes problem above with `w = β·u`. Large
residual → small weight. Weights are renormalised to mean 1 so the
problem scale (and backward compatibility) is preserved.

### 4.2 Node-level view attention (ext. 2)

Instead of one `α_v`, each node gets its own per-view weights from the
view's reconstruction quality at that node:

```
  e_{v,i} = ‖(H_v − T W_vᵀ)_i‖²,
  β_{v,i} = exp(−e_{v,i}/λ) / Σ_w exp(−e_{w,i}/λ).
```

A modality that explains node `i` well dominates the consensus *for
that node*, so different modalities can lead for different samples.
When `e_{v,i}` is constant across views it collapses to a global
weight, preserving the LMGEC special case.

### 4.3 Dimension-normalised attention (ext. 3)

Raw `e_{v,i}` scales with the view dimension `d_v` (a sum of `d_v`
squared residuals), biasing attention toward low-dimensional
modalities. We divide by `d_v`:
`ẽ_{v,i} = e_{v,i}/d_v`.

> **Design fix discovered empirically.** Naively using `ẽ = e/d` with a
> fixed `λ` flattens the softmax (the effective temperature blows up by
> `d`), neutralising attention. We therefore make `λ` a **unitless
> relative temperature** by dividing all errors by their global median
> before the softmax (`components.view_attention`). With this, dimension
> normalisation removes per-view dimension bias *without* destroying the
> per-node discrimination on equal-dim data.
>
> **Empirical verdict: discarded.** Even after this fix, the
> experimental report shows dimension normalisation brings *no upside*
> on equal-dim data and is *actively harmful* under extreme dimension
> heterogeneity (it over-promotes a degenerate 3-dim view on
> `prokaryotic`, collapsing NMI 0.27→0.11). It is therefore **off in
> the recommended configuration** (kept available as a toggle). See the
> experimental report, §3–4.

### 4.4 Static vs dynamic attention (ext. 4)

* **Static** — `β` computed once from the initial reconstruction and
  frozen (cheapest).
* **Dynamic** — `β` recomputed every iteration as `W,F` evolve
  (adapts to the refined latent space).

### 4.5 Attention regularisation (ext. 5)

The per-node weighting solves
`min_β Σ_v β_v e_v + λ Ω(β)` s.t. `β≥0, Σβ=1`:

* `Ω = Σ β log β` (**entropy**) → closed form `β = softmax(−e/λ)`;
  entropy regularisation is *exactly* temperature scaling.
* `Ω = Σ β²` (**L2**) → Euclidean **projection of `−e/2λ` onto the
  probability simplex** (sort-based), which can produce *sparse* weights
  (a modality can be switched off for a node).

### 4.6 Controlled graph propagation (ext. 6)

Replace `H_v = S_v X_v` by a convex blend

```
  H_v = ((1−γ_v) I + γ_v S_v) X_v,   γ_v ∈ [0,1].
```

`γ=1` ≡ LMGEC; `γ=0` ≡ raw features; intermediate `γ` mitigates
over-smoothing. `γ` may be a fixed grid value or **data-driven** from
the normalised graph-signal smoothness
`Smooth_v = tr(X_vᵀ L_v X_v)/‖X_v‖_F²`, `L_v = I−S_v`
(rougher signal → more propagation). Only affects views that have a
graph; on feature-only multimodal data it is a no-op.

### 4.7 Soft clustering (ext. 7)

Soft (soft-K-means) responsibilities
`R_{ic} = softmax_c(−β_s ‖Z_i − f_c‖²)` replace hard labels; centroids
become responsibility means and the target is `T = R F`. `β_s` controls
sharpness (`β_s→∞` recovers hard K-Means). Boundary nodes contribute to
several clusters.

### 4.8 Cross-view alignment (ext. 8)

Penalise per-view representation drift with `μ Σ_{v<w}‖Z_v−Z_w‖²`.
Operationally (keeping the Procrustes closed form) we augment each
view's target with the previous consensus,
`T_v = T + μ Z_prev`, pulling `Z_v` toward the shared space.

---

## 5. Optimisation procedure (full)

```
preprocess each view  → H_v   (controlled propagation if enabled)
m ← min(embedding_dim, min_v rank(H_v))            # rank-safe latent dim
init: per-view SVD + KMeans → a-priori α_v ; consensus Z ; (G,F) = KMeans(Z)
β ← α (broadcast) ; u ← 1 ; (static β computed once here if requested)
repeat max_iter times:
    T ← F[G]            (hard)   or   R F   (soft)
    for v:
        W_v ← Procrustes( H_vᵀ diag(β_v·u_v) (T + μ·Z_prev·[align]) )
        Z_v ← H_v W_v ;  e_v ← recon error ;  u_v ← IRLS weight [if robust]
    β ← attention(e ; λ, reg)      # if node_attention (dynamic);
                                   #   else static β, else global α
    Z ← Σ_v β_v ⊙ Z_v
    (G or R) ← assign(Z, F) ;  F ← weighted responsibility mean
    Z_prev ← Z
return labels (argmax R if soft), F, Z, β (attention_), loss history
```

---

## 6. Complexity analysis

Let `d = max_v d_v`. Per iteration and per view the dominant cost is
the Procrustes SVD of the `d_v×m` matrix `H_vᵀ(...)T`:
`O(n d_v m + d_v m²)`. The added components are all `≤` this order:

| Component | Extra cost / iter | Note |
|-----------|-------------------|------|
| robust IRLS `u` | `O(n d_v m)` | residual already computed |
| node attention `β` | `O(n V)` | softmax over `V` views |
| simplex projection (L2 reg) | `O(n V log V)` | sort of `V` entries |
| controlled propagation | `O(nnz(S_v) d_v)` | once, preprocessing only |
| soft assignment | `O(n k m)` | replaces `O(n k m)` argmin |
| alignment | `O(n m²)` | target augmentation |

**Total:** `O( Σ_v (n d_v m + d_v m²) )` per iteration — the **same
asymptotic order as LMGEC**, with a small constant overhead (empirically
~1.3–2× wall-clock, mainly the dense node-weighted Procrustes).
Additional memory: one `n×k` responsibility matrix (soft) and one
`V×n` attention matrix.

---

## 7. Hyperparameters

| Name | Default | Meaning |
|------|---------|---------|
| `embedding_dim` | 16 | latent dim (auto-clamped to view rank) |
| `temperature` | 1.0 | global a-priori weight temperature; default attention λ |
| `robust_loss` | False | ℓ2,1 (IRLS) loss |
| `node_attention` | False | node-level attention |
| `dim_normalized_attention` | False | de-bias attention by view dimension |
| `dynamic_attention` | True | recompute attention each iter (vs static) |
| `attention_regularization` | None | `None`/`"entropy"`/`"l2"` |
| `attention_lambda` | None (=temperature) | attention strength λ |
| `controlled_propagation` | False | partial propagation |
| `gamma` | 1.0 / `"auto"` | propagation strength γ |
| `soft_clustering` | False | soft K-means assignments |
| `soft_beta` | 50.0 | soft assignment sharpness (low values collapse!) |
| `cross_view_alignment` | False | representation alignment |
| `align_mu` | 0.1 | alignment strength μ |
| `beta`,`tf_idf`,`center`,`scale`,`preprocess`,`max_iter` | (LMGEC) | shared |

The empirically-recommended configuration is given in the experimental
report; in short, **node attention (dynamic) is the robust default**,
with the other extensions kept off unless the data calls for them.
