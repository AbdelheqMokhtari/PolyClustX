<p align="center"><img src="https://raw.githubusercontent.com/AbdelheqMokhtari/PolyClustX/v1.4.1/src/polyclustx/display/assets/logo.png" width="150" alt="PolyClustX logo"></p>

<p align="center">
  <a href="https://pypi.org/project/polyclustx/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/polyclustx?color=4E4E92"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-557AA3?logo=python&logoColor=white">
  <a href="https://github.com/AbdelheqMokhtari/PolyClustX/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/AbdelheqMokhtari/PolyClustX/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/AbdelheqMokhtari/PolyClustX/blob/v1.4.1/LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-634E87"></a>
  <a href="https://polyclustx.readthedocs.io/en/latest/"><img alt="Documentation" src="https://img.shields.io/badge/docs-Read%20the%20Docs-4E4E92"></a>
</p>

# PolyClustX

**A Unified Framework for Multi-View and Multimodal Representation Learning and Clustering from Feature Views to Attributed Graphs**

Prepare data, compare algorithms, tune hyperparameters and visualize results
through one Python API, command line or graphical interface. Work with one or
many feature views, optional graphs and optional ground-truth labels.

[Documentation](https://polyclustx.readthedocs.io/en/latest/) ·
[Models](https://polyclustx.readthedocs.io/en/latest/background/algorithms.html) ·
[Examples](https://polyclustx.readthedocs.io/en/latest/user_guide/examples.html)

## Install

Python **3.10+** is required. Installation includes PyTorch, the interface and
Optuna; Hugging Face datasets use ordinary HTTP downloads.

From PyPI:

```bash
python -m pip install polyclustx
```

Or install the repository:

```bash
git clone https://github.com/AbdelheqMokhtari/PolyClustX.git
cd PolyClustX
python -m pip install .
```

Choose how you want to work:
[Interface](#1-interactive-interface) · [Python](#2-python-package) · [CLI](#3-command-line).

## 1. Interactive interface

```bash
polyclustx-display
```

Open the printed address, normally http://localhost:8501.
If the launcher is not found, use `python -m polyclustx.display`.

Load a dataset, inspect its features/graphs, and compose named views. Prepare
models, run comparisons or searches, visualize results and download reports
from the independent pages. Variant controls appear only for datasets with
alternate representations.
[Interface guide](https://polyclustx.readthedocs.io/en/latest/user_guide/workbench.html).

## 2. Python package

Compare two methods on two feature views of a small offline dataset:

```python
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from polyclustx.data import MultiViewDataset
from polyclustx.models import LMGEC, MultiViewKMeans
from polyclustx.metrics import clustering_accuracy, nmi_score
from polyclustx.experiment import Experiment

X, labels = load_wine(return_X_y=True)  # small, offline example
X = StandardScaler().fit_transform(X)
data = MultiViewDataset([X[:, :6], X[:, 6:]], labels=labels, name="Wine")

results = Experiment(
    datasets=data,
    models=[
        LMGEC(n_clusters=3, embedding_dim=3, max_iter=20, random_state=42),
        MultiViewKMeans(n_clusters=3, random_state=42),
    ],
    metrics={"ACC": clustering_accuracy, "NMI": nmi_score},
    runs=3,
    save_path="experiments/comparison",
).run()
```

`run()` prints a model-configuration table followed by a results table for each
dataset. No extra `print` is needed. Example result output:

```text
Results — Wine
┌────────┬──────────────────┬────────────────────┐
│ Metric │            LMGEC │    MultiViewKMeans │
├────────┼──────────────────┼────────────────────┤
│ ACC    │ 0.9551 (±0.0000) │ 0.9663 (±0.0000) * │
│ NMI    │ 0.8352 (±0.0000) │ 0.8770 (±0.0000) * │
│ Alphas │   [0.481, 0.519] │                    │
│ Time   │            0.03s │              0.02s │
└────────┴──────────────────┴────────────────────┘
```

Values are means and standard deviations over three runs; `*` marks the best
mean. Timing and numerical results can vary between environments. Tables adapt
to terminal width and remain scrollable in notebooks. The JSON/CSV reports are
saved separately; to download one bundle:

```python
results.export("experiments/comparison.zip")
```

Use `load_data("acm")` instead of constructing `data` to work with a catalog
dataset. For several datasets, pass `datasets={"Study A": data_a, "Study B": data_b}`;
prepare estimators with the K needed for your study.

For a learned embedding and a plot, fit a model directly:

```python
from polyclustx.visualization import plot_clusters

model = data.fit(LMGEC(n_clusters=3, embedding_dim=3, random_state=42))
plot_clusters(model.embedding_, model.labels_, method="pca")
```

[Experiment guide](https://polyclustx.readthedocs.io/en/latest/user_guide/experiments.html).

## 3. Command line

```bash
polyclustx models
polyclustx datasets
lmgec --help
lmgec acm --embedding-dim 8 --runs 3
scm acm --embedding-dim 16 --max-iter 50
idec acm --hidden-dims '128,64' --pretrain-epochs 20
```

Each method has its own command and parameter help. Use `--variant` for a dataset
variant, `--clusters` to override K, and `--output` to select the report directory.
Without an explicit K, labeled datasets use their number of reference classes.

Commands show live progress by default: the dataset, model, run or search trial,
and training-stage updates when the method reports them. Use `--verbose 1` for
tables only, `--verbose 0` to hide fitting progress and tables, or `--verbose 3`
for detailed logs. The short form is `-v 2`:

```bash
lmgec acm --embedding-dim 8 --runs 3 --verbose 2
polyclustx experiment --config examples/experiment_plan.yaml -v 2
```

Search candidates or continuous ranges:

```bash
lmgec acm --search grid --vary 'temperature=[0.5,1,2]' --objective nmi
lmgec acm --search optuna --vary 'temperature={"low":0.1,"high":5.0}' --trials 10
multiview-kmeans acm --no-benchmark --clusters 3 --metrics silhouette db
```

Compare several methods and datasets in **one report**:

```bash
polyclustx experiment --datasets acm dblp --models lmgec scm --runs 3
polyclustx experiment --datasets acm --models lmgec scm --param lmgec.embedding_dim=8
polyclustx experiment --config examples/experiment_plan.yaml
```

The [example plan](https://github.com/AbdelheqMokhtari/PolyClustX/blob/main/examples/experiment_plan.yaml) keeps larger studies readable:
name configurations, set per-model parameters, choose datasets and specify repeats.
Dataset-specific default K is handled automatically. Short commands may conflict
with other installed tools; use a dedicated environment or `polyclustx lmgec acm`.

**Interface:** `polyclustx-display` is a standalone command.
See the [CLI guide with complete examples](https://polyclustx.readthedocs.io/en/latest/api/cli.html).

## Datasets

37 dataset families provide 66 MATLAB representation files. All download links
are direct Hugging Face URLs: no extra download package or account is needed.

| Category | What the data contains |
| --- | --- |
| MVAGC | Same-modality attributes and one or more structural graphs |
| MVC | Several feature representations of the same modality; no graphs |
| MMC | Different modalities, such as images and text; no graphs |
| MMAGC | Multimodal node features and graph relationships |

Missing observations are independent of these categories. Simulate them with
`load_data("acm", 0.2, random_state=42)`, or align your own views using sample IDs.
Different methods support different inputs; see the method table below.

### Multi-view attributed graph clustering (MVAGC)

| Dataset | Samples / nodes | Classes | Stored feature matrices | Graphs | Download | Reference / background |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `ACM` | 3,025 | 3 | 1 | 2 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/ACM/ACM.mat?download=true) | [Reference](https://github.com/chakib401/LMGEC) |
| `Amazon_photos` | 7,487 | 8 | 1 | 1 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/Amazon_photos/Amazon_photos.mat?download=true) | [Reference](https://github.com/chakib401/LMGEC) |
| `DBLP` | 4,057 | 4 | 1 | 3 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/DBLP/DBLP.mat?download=true) | [Reference](https://github.com/chakib401/LMGEC) |
| `IMDB` | 4,780 | 3 | 1 | 2 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/IMDB/IMDB.mat?download=true) | [Reference](https://github.com/chakib401/LMGEC) |
| `Wiki` | 2,405 | 17 | 1 | 1 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/Wiki/Wiki.mat?download=true) | [Reference](https://github.com/chakib401/LMGEC) |

### Multi-view clustering (MVC)

| Dataset | Samples / nodes | Classes | Stored feature matrices | Graphs | Download | Reference / background |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `100leaves` | 1,600 | 100 | 3 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/100leaves/100leaves.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `3sources` | 169 | 6 | 3 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/3sources/3sources.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `aloi` | 10,800 | 100 | 4 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/aloi/aloi.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `bbcsport` | 544 | 5 | 2 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/bbcsport/bbcsport.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `caltech101-7` | 1,474 | 7 | 6 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/caltech101-7/caltech101-7.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `handwritten` | 2,000 | 10 | 6 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/handwritten/handwritten.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `mnist-10k` | 10,000 | 10 | 3 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/mnist-10k/mnist-10k.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `mnist-4` | 4,000 | 4 | 3 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/mnist-4/mnist-4.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `msrc-v5` | 210 | 7 | 5 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/msrc-v5/msrc-v5.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `outdoorscence` | 2,688 | 8 | 4 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/outdoorscence/outdoorscence.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `proteinfold` | 694 | 27 | 12 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/proteinfold/proteinfold.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `reuters-1200` | 1,200 | 6 | 5 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/reuters-1200/reuters-1200.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `reuters-1500` | 1,500 | 6 | 5 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/reuters-1500/reuters-1500.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `uci` | 2,000 | 10 | 3 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/uci/uci.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `webkb` | 1,051 | 2 | 2 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/webkb/webkb.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `yale` | 165 | 15 | 3 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/yale/yale.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |

### Multimodal clustering (MMC)

| Dataset | Samples / nodes | Classes | Stored feature matrices | Graphs | Download | Reference / background |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `coco-all` | 23,189 | 43 | 2 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/coco-all/coco-all.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `coco-cross` | 7,429 | 10 | 2 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/coco-cross/coco-cross.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `esp-game` | 11,032 | 7 | 2 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/esp-game/esp-game.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `iapr-tc12` | 7,855 | 6 | 2 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/iapr-tc12/iapr-tc12.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `mirflickr` | 12,154 | 6 | 2 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/mirflickr/mirflickr.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `nus-wide` | 20,000 | 8 | 2 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/nus-wide/nus-wide.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `pascal` | 1,000 | 20 | 2 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/pascal/pascal.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |
| `wikipedia-test` | 693 | 10 | 2 | 0 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/wikipedia-test/wikipedia-test.mat?download=true) | [Reference](https://github.com/ChuanbinZhang/Multi-view-datasets) |

### Multimodal attributed graph clustering (MMAGC)

| Dataset | Samples / nodes | Classes | Stored feature matrices | Graphs | Download | Reference / background |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `Arts` | 58,487 | 11 | 2 | 1 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/Arts/Arts.mat?download=true) | [Reference](https://github.com/sktsherlock/MAGB) |
| `grocery` | 84,379 | 20 | 2 | 1 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/grocery/grocery.mat?download=true) | [Reference](https://github.com/sktsherlock/MAGB) |
| `grocery-S` | 17,074 | 20 | 2 | 1 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/grocery-S/grocery-S.mat?download=true) | [Reference](https://github.com/sktsherlock/MAGB) |
| `movies` | 16,672 | 20 | 2 | 1 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/movies/movies.mat?download=true) | [Reference](https://github.com/sktsherlock/MAGB) |
| `photo` | 36,731 | 12 | 2 | 1 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/photo/photo.mat?download=true) | [Reference](https://github.com/sktsherlock/MAGB) |
| `reddit` | 99,638 | 50 | 2 | 1 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/reddit/reddit.mat?download=true) | [Reference](https://github.com/sktsherlock/MAGB) |
| `reddit-s` | 15,894 | 20 | 2 | 1 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/reddit-s/reddit-s.mat?download=true) | [Reference](https://github.com/sktsherlock/MAGB) |
| `toys` | 20,695 | 18 | 2 | 1 | [MAT](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/toys/toys.mat?download=true) | [Reference](https://github.com/sktsherlock/MAGB) |

### Alternate feature representations

Variants reuse the same samples with different encoders, rather than adding
new families. Select one with `load_data("pascal", variant="clip")`.
Grocery-S and Reddit-S are separate smaller datasets, each with its own variants.

| Dataset family | Additional representations: download and stored image/text widths |
| --- | --- |
| `Arts` | [clip (768/768)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/Arts/Arts_Clip.mat?download=true) |
| `coco-all` | [clip (512/512)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/coco-all/coco-all_Clip.mat?download=true) · [resnet_minilm (2048/384)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/coco-all/coco-all_resnet_minilm.mat?download=true) · [vit_minilm (768/384)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/coco-all/coco-all_vit_minilm.mat?download=true) · [vit_sbert (768/768)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/coco-all/coco-all_vit_sbert.mat?download=true) |
| `coco-cross` | [clip (512/512)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/coco-cross/coco-cross_Clip.mat?download=true) · [resnet_minilm (2048/384)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/coco-cross/coco-cross_resnet_minilm.mat?download=true) · [vit_minilm (768/384)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/coco-cross/coco-cross_vit_minilm.mat?download=true) · [vit_sbert (768/768)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/coco-cross/coco-cross_vit_sbert.mat?download=true) |
| `grocery` | [clip (768/768)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/grocery/grocery_Clip.mat?download=true) |
| `grocery-S` | [clip (768/768)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/grocery-S/grocery-S_Clip.mat?download=true) · [clip_minilm (512/384)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/grocery-S/grocery-S_Clip_minilm.mat?download=true) · [resnet_clip (2048/512)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/grocery-S/grocery-S_resnet_Clip.mat?download=true) · [resnet_minilm (2048/384)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/grocery-S/grocery-S_resnet_minilm.mat?download=true) |
| `movies` | [clip (768/768)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/movies/movies_Clip.mat?download=true) · [clip_minilm (512/384)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/movies/movies_Clip_minilm.mat?download=true) · [resnet_clip (2048/512)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/movies/movies_resnet_Clip.mat?download=true) · [resnet_minilm (2048/384)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/movies/movies_resnet_minilm.mat?download=true) |
| `pascal` | [clip (512/512)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/pascal/pascal_Clip.mat?download=true) · [resnet_minilm (2048/384)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/pascal/pascal_resnet_minilm.mat?download=true) · [vit_minilm (768/384)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/pascal/pascal_vit_minilm.mat?download=true) · [vit_sbert (768/768)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/pascal/pascal_vit_sbert.mat?download=true) |
| `photo` | [clip (768/768)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/photo/photo_Clip.mat?download=true) |
| `reddit` | [clip (768/768)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/reddit/reddit_Clip.mat?download=true) |
| `reddit-s` | [clip (768/768)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/reddit-s/reddit-s_Clip.mat?download=true) |
| `toys` | [clip (768/768)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/toys/toys_Clip.mat?download=true) · [clip_minilm (512/384)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/toys/toys_Clip_minilm.mat?download=true) · [resnet_clip (2048/512)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/toys/toys_resnet_Clip.mat?download=true) · [resnet_minilm (2048/384)](https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/toys/toys_resnet_minilm.mat?download=true) |

Pascal and COCO default to ResNet + Doc2Vec (2048/300 columns).
MMAGC defaults to CLIP-ViT + Gemma-7B (768/3072 columns).
ESP-Game, IAPR-TC12, MIRFlickr and NUS-WIDE have VGG-16/BERT source
representations but store processed 100/100-column views in the supplied MAT files.
Actual shapes are always available in `data.feature_dims` and the interface.
Photo/Arts origins and Grocery-S subset construction remain undocumented;
reference links provide background rather than verified provenance for every processed artifact.

See [dataset descriptions, graph meanings and all variants](https://polyclustx.readthedocs.io/en/latest/catalog/datasets.html).
If an upload is still pending, its loader reports the unavailable file; the
same download link becomes usable once published.

### Load data and choose views

```python
from polyclustx.data import load_data, load_mat

data = load_data("acm", views={
    "Co-authorship": ("X1_features", "A1_PAP"),
    "Shared subjects": ("X1_features", "A2_PLP"),
    "Both relations": ("X1_features", ["A1_PAP", "A2_PLP"]),
    "Keywords only": "X1_features",
})
# The same feature is reused; each view has the graphs you selected.

captions = load_data("pascal", variant="clip")
my_data = load_mat("my_data.mat")  # recognizes X*, A* and labels
```

One named entry creates a single view; several graphs in one entry are fused
at fitting/preparation time. Source files and previous experiment results stay
unchanged. [Loading and composition guide](https://polyclustx.readthedocs.io/en/latest/user_guide/data.html).

## Available methods

Graphs ✓ means the method uses supplied adjacency matrices. Missing views ✓
means native support for an observation mask.

| Category | Method / command | Graphs | Missing views |
|---|---|:---:|:---:|
| Graph-aware | LMGEC / `lmgec` | ✓ | — |
| Graph-aware | DGF / `dgf` | ✓ | — |
| Graph-aware | SGLA / `sgla` | ✓ | — |
| Feature-based | MultiViewKMeans / `multiview-kmeans` | — | — |
| Feature-based | MultiViewSpectralClustering / `multiview-spectral` | — | — |
| Feature-based | G-CoMVKM / `g-comvkm` | — | — |
| Feature-based | IDEC / `idec` | — | — |
| Feature-based | CONAN / `conan` | — | — |
| Feature-based | SCM / `scm` | — | — |
| Feature-based | MFLVC / `mflvc` | — | — |
| Feature-based | UMC / `umc` | — | — |
| Incomplete-view | COMPLETER / `completer` | — | ✓ |
| Incomplete-view | DCP / `dcp` | — | ✓ |

Capabilities differ by method. Some single-view and multi-view cases are
framework adaptations; UMC uses supplied features without pretrained encoders.
See [algorithm details, assumptions and references](https://polyclustx.readthedocs.io/en/latest/user_guide/clustering_models.html).
These implementations do not claim to reproduce the papers' benchmark scores.

## Explore and visualize

The interface provides dataset preparation, named models, repeated experiments,
hyperparameter search, visualizations and JSON/CSV/ZIP exports.
Labels enable benchmark metrics; without labels, use internal metrics.
[Interface guide](https://polyclustx.readthedocs.io/en/latest/user_guide/workbench.html).

Run the [reference notebook](https://github.com/AbdelheqMokhtari/PolyClustX/blob/main/notebooks/01_lmgec_experiments_and_optuna.ipynb)
to list models and datasets, compare three LMGEC configurations on ACM and
IMDB, tune LMGEC with Optuna, visualize embeddings and losses, and export
results. The notebook is saved **without outputs** so you can run it yourself.

## Documentation

Browse the [documentation](https://polyclustx.readthedocs.io/en/latest/),
start with the [worked examples](https://polyclustx.readthedocs.io/en/latest/user_guide/examples.html), or follow the
[interactive interface guide](https://polyclustx.readthedocs.io/en/latest/user_guide/workbench.html).

## Research affiliations

<p>
  <a href="https://centreborelli.ens-paris-saclay.fr/fr"><img src="https://raw.githubusercontent.com/AbdelheqMokhtari/PolyClustX/v1.4.1/src/polyclustx/display/assets/borelli.png" height="90" alt="Centre Borelli"></a>
  &nbsp;&nbsp;
  <a href="https://u-paris.fr/"><img src="https://raw.githubusercontent.com/AbdelheqMokhtari/PolyClustX/v1.4.1/src/polyclustx/display/assets/paris_cite.png" height="90" alt="Université Paris Cité"></a>
</p>

Abdelheq Mokhtari · Lazhar Labiod · Mohamed Nadif.
All authors are affiliated with Centre Borelli, Université Paris Cité.
Institutional marks remain separate from PolyClustX and do not imply endorsement.

## References

Cite the original papers when using these methods in research. Reference code
links identify the implementations that informed the framework adaptations.

| Method | Paper | Reference implementation |
|---|---|---|
| LMGEC | [Fettal et al., WSDM 2023](https://doi.org/10.1145/3539597.3570367) | [Code](https://github.com/chakib401/LMGEC) |
| DGF | [Zheng et al., Dual Graph Filtering](https://arxiv.org/abs/2511.20030) | [Code](https://github.com/HaoranZ99/DGF) |
| SGLA | [Li et al., ICDE 2025](https://arxiv.org/abs/2508.09452) | [Code](https://github.com/CyanideCentral/SGLA) |
| MultiViewKMeans | [Bickel & Scheffer, ICDM 2004](https://doi.org/10.1109/ICDM.2004.10095) | [Code](https://github.com/mvlearn/mvlearn) |
| MultiViewSpectralClustering | [Kumar & Daumé, ICML 2011](https://icml.cc/2011/papers/272_icmlpaper.pdf) | [Code](https://github.com/mvlearn/mvlearn) |
| G-CoMVKM | [Sinaga & Yang, Electronics 2025](https://doi.org/10.3390/electronics14112129) | [Code](https://github.com/KristinaP09/G-CoMVKM) |
| IDEC | [Guo et al., IJCAI 2017](https://www.ijcai.org/proceedings/2017/243) | [Code](https://github.com/XifengGuo/IDEC) |
| CONAN | [Ke et al., BigData 2021](https://doi.org/10.1109/BigData52589.2021.9671851) | [Code](https://github.com/Guanzhou-Ke/conan) |
| SCM | [Luo et al., IJCAI 2024](https://www.ijcai.org/proceedings/2024/519) | [Code](https://github.com/SubmissionsIn/SCM) |
| MFLVC | [Xu et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_Multi-Level_Feature_Learning_for_Contrastive_Multi-View_Clustering_CVPR_2022_paper.html) | [Code](https://github.com/SubmissionsIn/MFLVC) |
| UMC | [Zhang et al., ACL 2024](https://aclanthology.org/2024.acl-long.2/) | [Code](https://github.com/thuiar/UMC) |
| COMPLETER | [Lin et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Lin_COMPLETER_Incomplete_Multi-View_Clustering_via_Contrastive_Prediction_CVPR_2021_paper.pdf) | [Code](https://github.com/XLearning-SCU/2021-CVPR-Completer) |
| DCP | [Lin et al., TPAMI](https://doi.org/10.1109/TPAMI.2022.3197238) | [Code](https://github.com/XLearning-SCU/2022-TPAMI-DCP) |


## License

The software license and third-party notices are in [LICENSE](https://github.com/AbdelheqMokhtari/PolyClustX/blob/v1.4.1/LICENSE). Institutional logos are not
covered by the software's MIT license. Cite the original algorithm papers
when publishing results.
