# pymv notebooks

Hands-on walkthrough of the library, one stage at a time. Every notebook
runs **offline in a few seconds** on a small synthetic multi-view dataset
defined in its first cell — swap in a real benchmark by uncommenting the
`load_dataset(...)` line.

| # | Notebook | Stage |
|---|----------|-------|
| 00 | [`00_quickstart.ipynb`](00_quickstart.ipynb) | the whole pipeline at a glance |
| 01 | [`01_data_and_preprocessing.ipynb`](01_data_and_preprocessing.ipynb) | the dataset catalog + normalize → propagate |
| 02 | [`02_lmgec_baseline.ipynb`](02_lmgec_baseline.ipynb) | the original LMGEC algorithm, step by step |
| 03 | [`03_lmgecplus_components.ipynb`](03_lmgecplus_components.ipynb) | swapping LMGECplus components + custom ones |
| 04 | [`04_ablation_study.ipynb`](04_ablation_study.ipynb) | one-call ablation ladder + random-search tuning |

## Running

```bash
pip install -e .
jupyter lab            # then open the notebooks in order
```

Or execute the whole set headless:

```bash
for nb in notebooks/0*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

The two models under study:

- **`LMGEC`** — the original Linear Multi-view Graph Embedding & Clustering
  algorithm (TensorFlow).
- **`LMGECplus`** — a modular re-implementation where every component
  (loss, normalization, propagation, attention/weighting, alignment,
  clustering head, optimizer) is swappable. With no overrides it reproduces
  LMGEC exactly, which is what makes clean ablations possible.
