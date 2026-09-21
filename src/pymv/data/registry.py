"""
Dataset catalog and registry.

Every dataset is described by a single :class:`DatasetSpec` entry in
``_SPECS`` — its download URL, cache location, file layout (reader),
and a short description.  Adding a new dataset means adding **one
entry** there; the ``load_<name>`` function, ``load_dataset``,
``list_datasets`` and the CLI all pick it up automatically.

Public API
----------
- :func:`load_dataset` — load any dataset by name.
- :func:`list_datasets` — list available dataset names.
- :func:`get_dataset_info` — metadata (views, category, cache status)
  without downloading anything.
- ``load_<name>`` — one convenience function per dataset
  (e.g. ``load_acm()``), re-exported from :mod:`pymv.data`.
- ``DATASETS`` — mapping of ``name -> zero-argument loader``.
"""

from __future__ import annotations

import difflib
import logging
import math
import warnings
from dataclasses import dataclass, field
from numbers import Real
from typing import Callable

import scipy.io as io

from . import readers
from .base import get_dataset_path
from .dataset import MultiViewDataset
from .downloader import download_file

logger = logging.getLogger(__name__)

# Zenodo records hosting the .mat files
_R_ORIGINAL = "18923707"
_R_RETRIEVAL = "19470217"
_R_BENCHMARK = "20732535"
_REFERENCE_COLLECTION = "https://github.com/ChuanbinZhang/Multi-view-datasets"
# Published collection statistics, not a promise about transformed hosted artifacts.
_REFERENCE_DETAILS = {
    "3sources": "169 news stories; 6 classes; Reuters/BBC/Guardian dimensions 3068/3560/3631.",
    "100leaves": "1,600 leaf images; 100 species; three 64-dimensional descriptors.",
    "bbcsport": "544 sports documents; 5 classes; feature dimensions 3183/3203.",
    "caltech101_7": "1,474 images; 7 classes; six descriptors with dimensions 48/40/254/1984/512/928.",
    "uci": "2,000 digit images; 10 classes; feature dimensions 240/76/6.",
    "yale": "165 face images; 15 subjects; feature dimensions 4096/3304/6750.",
    "reuters_1200": "1,200 documents; 6 classes; five languages with 2,000 features each.",
}


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to download and read one dataset."""

    name: str                 # registry key, e.g. "acm"
    record: str               # Zenodo record id
    filename: str             # local cache filename
    reader: Callable          # readers.* function: (data, **kwargs) -> (Xs, As, labels)
    kwargs: dict = field(default_factory=dict)
    category: str = "features"    # "features" | "graph" | "multimodal"
    n_views: int = 2
    description: str = ""
    cache_dir: str | None = None  # defaults to `name`
    remote: str | None = None     # remote filename, defaults to `filename`
    reference: str = _REFERENCE_COLLECTION
    reference_verified: bool = False
    reference_details: str = ""
    modalities: tuple[str, ...] | None = None

    @property
    def url(self) -> str:
        remote = self.remote or self.filename
        return f"https://zenodo.org/records/{self.record}/files/{remote}?download=1"

    @property
    def path(self):
        return get_dataset_path(self.cache_dir or self.name) / self.filename


def _standard(name, filename, n_views, description=""):
    """Spec for the common cell-array layout (``X`` cells + ``y``)."""
    return DatasetSpec(
        name, _R_BENCHMARK, filename, readers.read_standard,
        kwargs={"n_views": n_views}, n_views=n_views,
        description=description,
        reference_verified=True,
        reference_details=_REFERENCE_DETAILS.get(name, ""),
    )


def _multimodal(name, filename, description=""):
    """Spec for the image/text + shared-graph layout."""
    return DatasetSpec(
        name, _R_BENCHMARK, filename, readers.read_multimodal,
        category="multimodal", n_views=2, description=description,
    )


def _image_text(name, filename, description="", record=_R_RETRIEVAL):
    """Spec for two-view retrieval sets stored as ``img``/``txt``/``label``."""
    return DatasetSpec(
        name, record, filename, readers.read_feature_views,
        kwargs={"keys": ("img", "txt"), "label_key": "label"},
        n_views=2, description=description,
    )


def _x1_x2(name, filename, description="", record=_R_ORIGINAL, cache_dir=None):
    """Spec for two-view sets stored as ``X1``/``X2``/``Y``."""
    return DatasetSpec(
        name, record, filename, readers.read_feature_views,
        kwargs={"keys": ("X1", "X2"), "label_key": "Y"},
        n_views=2, description=description, cache_dir=cache_dir,
    )


def _embedding_variants(base, description):
    """The CLIP / ResNet / ViT / SBERT embedding variants of a dataset."""
    return [
        _x1_x2(f"{base}_{v}", f"{base}_{v}.mat", f"{description} ({v} embeddings)",
               record=_R_RETRIEVAL, cache_dir=base)
        for v in ("clip", "resnet_minilm", "vit_minilm", "vit_sbert")
    ]


_SPECS = [
    # ── Graph datasets (features + one adjacency per view) ────────
    DatasetSpec(
        "acm", _R_ORIGINAL, "ACM.mat", readers.read_graph,
        kwargs={"feature_key": "features", "adj_keys": ("PAP", "PLP"),
                "label_key": "label"},
        category="graph", n_views=2,
        description="ACM papers — co-author (PAP) and co-subject (PLP) graphs",
    ),
    DatasetSpec(
        "dblp", _R_ORIGINAL, "DBLP.mat", readers.read_graph,
        kwargs={"feature_key": "features",
                "adj_keys": ("net_APTPA", "net_APCPA", "net_APA"),
                "label_key": "label"},
        category="graph", n_views=3,
        description="DBLP authors — three meta-path graphs",
    ),
    DatasetSpec(
        "imdb", _R_ORIGINAL, "IMDB.mat", readers.read_graph,
        kwargs={"feature_key": "features", "adj_keys": ("MAM", "MDM"),
                "label_key": "label"},
        category="graph", n_views=2,
        description="IMDB movies — co-actor (MAM) and co-director (MDM) graphs",
    ),
    DatasetSpec(
        "amazon_photos", _R_ORIGINAL, "amazon_photos.mat",
        readers.read_amazon_photos, remote="Amazon_photos.mat",
        category="graph", n_views=2,
        description="Amazon photos co-purchase graph — raw + similarity features",
    ),
    DatasetSpec(
        "wiki", _R_ORIGINAL, "wiki.mat", readers.read_wiki,
        category="graph", n_views=2,
        description="Wikipedia articles — link graph + kNN graph views",
    ),

    # ── Image–text feature datasets ────────────────────────────────
    _x1_x2("pascal", "pascal.mat", "Pascal VOC image + text embeddings"),
    *_embedding_variants("pascal", "Pascal VOC image + text"),
    _x1_x2("coco_all", "coco_all.mat", "MS-COCO (all) image + caption embeddings"),
    *_embedding_variants("coco_all", "MS-COCO (all) image + caption"),
    _x1_x2("coco_cross", "coco_cross.mat", "MS-COCO (cross) image + caption embeddings"),
    *_embedding_variants("coco_cross", "MS-COCO (cross) image + caption"),
    _x1_x2("bdgp", "BDGP.mat", "Drosophila embryo images — visual + textual features"),
    _image_text("esp_game", "esp-game.mat", "ESP-Game images + tags"),
    _image_text("flickr30k", "flickr30k.mat", "Flickr30k images + captions"),
    _image_text("iaprtc12", "iaprtc12.mat", "IAPR TC-12 images + captions"),
    _image_text("nus_wide", "nus-wide.mat", "NUS-WIDE images + tags"),
    DatasetSpec(
        "prokaryotic", _R_ORIGINAL, "Prokaryotic.mat",
        readers.read_feature_views, remote="prokaryotic.mat",
        kwargs={"keys": ("gene_repert", "text", "proteome_comp"),
                "label_key": "truth"},
        n_views=3,
        description="Prokaryotic species — gene repertoire, text, proteome",
    ),

    # ── Standard multi-view benchmarks (X cell array + y) ──────────
    _standard("100leaves", "100Leaves.mat", 3, "100 plant species — shape/texture/margin"),
    _standard("3sources", "3Sources.mat", 3, "News stories from BBC, Guardian, Reuters"),
    _standard("acm_5view", "ACM.mat", 5, "ACM papers — 5 feature views"),
    _standard("aloi", "ALOI.mat", 4, "ALOI object images — 4 feature views"),
    _standard("bbc4view", "BBC4view.mat", 4, "BBC news articles split into 4 segments"),
    _standard("bbcsport", "BBCSport.mat", 2, "BBC sport articles split into 2 segments"),
    _standard("caltech101_7", "Caltech101-7.mat", 6, "Caltech-101 (7 classes) — 6 feature types"),
    _standard("citeseer", "CiteSeer.mat", 2, "CiteSeer citation network features"),
    _standard("cora", "Cora.mat", 4, "Cora citation network features"),
    _standard("handwritten", "Handwritten.mat", 6, "UCI handwritten digits — 6 feature sets"),
    _standard("mnist_10k", "MNIST-10k.mat", 3, "MNIST subset (10k) — 3 feature views"),
    _standard("mnist_4", "MNIST-4.mat", 3, "MNIST (4 digits) — 3 feature views"),
    _standard("msrc_v5", "MSRC-v5.mat", 5, "MSRC-v5 images — 5 feature views"),
    _standard("outdoor_scene", "OutdoorScene.mat", 4, "Outdoor scene images — 4 feature views"),
    _standard("protein_fold", "ProteinFold.mat", 12, "Protein fold prediction — 12 views"),
    _standard("reuters_1200", "Reuters-1200.mat", 5, "Multilingual Reuters subset (1200 docs)"),
    _standard("reuters_1500", "Reuters-1500.mat", 5, "Multilingual Reuters subset (1500 docs)"),
    _standard("uci", "UCI.mat", 3, "UCI digits — 3 feature views"),
    _standard("webkb", "WebKB.mat", 2, "WebKB university web pages"),
    _standard("wikipedia", "Wikipedia-test.mat", 2, "Wikipedia articles — image + text features"),
    _standard("yale", "Yale.mat", 3, "Yale faces — 3 feature views"),

    # ── Multimodal (image + text embeddings with shared graph) ─────
    _multimodal("arts_multimodal", "Arts_clip_multimodal.mat", "Amazon Arts — CLIP image + text with graph"),
    _multimodal("grocery_multimodal", "Grocery_multimodal.mat", "Amazon Grocery — image + text with graph"),
    _multimodal("grocerys_multimodal", "GroceryS_multimodal.mat", "Amazon Grocery (small) — image + text with graph"),
    _multimodal("movies_multimodal", "Movies_multimodal.mat", "Movies — image + text with graph"),
    _multimodal("photo_multimodal", "Photo_multimodal.mat", "Amazon Photo — image + text with graph"),
    _multimodal("reddit_multimodal", "Reddit_multimodal.mat", "Reddit posts — image + text with graph"),
    _multimodal("reddits_multimodal", "RedditS_multimodal.mat", "Reddit posts (small) — image + text with graph"),
    _multimodal("toys_multimodal", "Toys_multimodal.mat", "Amazon Toys — image + text with graph"),
]

CATALOG: dict[str, DatasetSpec] = {spec.name: spec for spec in _SPECS}

# Public families resolve to physical artifacts. Existing artifact names remain
# accepted aliases, so old experiment files remain loadable.
VARIANTS = {name: {"default": name} for name in CATALOG}
for _base in ("pascal", "coco_all", "coco_cross"):
    for _variant in ("clip", "resnet_minilm", "vit_minilm", "vit_sbert"):
        _artifact = f"{_base}_{_variant}"
        VARIANTS[_base][_variant] = _artifact
        VARIANTS.pop(_artifact)
for _base, _variant, _artifact in (
    ("acm", "5view", "acm_5view"),
    ("grocery_multimodal", "small", "grocerys_multimodal"),
    ("reddit_multimodal", "small", "reddits_multimodal"),
):
    VARIANTS[_base][_variant] = _artifact
    VARIANTS.pop(_artifact)
ALIASES = {artifact: (base, variant) for base, variants in VARIANTS.items()
           for variant, artifact in variants.items() if artifact != base}


def _resolve(name, variant=None):
    key = str(name).lower().strip()
    if key in ALIASES:
        if variant not in (None, "default"):
            raise ValueError("Do not combine a legacy variant name with variant=")
        base, chosen = ALIASES[key]
        warnings.warn(f"Use load_dataset({base!r}, variant={chosen!r}) instead of {key!r}",
                      DeprecationWarning, stacklevel=3)
        return base, chosen, CATALOG[key]
    if key not in VARIANTS:
        close = difflib.get_close_matches(key, VARIANTS, n=3, cutoff=0.6)
        raise ValueError(f"Unknown dataset '{name}'. Did you mean {close}? Available: {sorted(VARIANTS)}")
    chosen = "default" if variant is None else str(variant).lower().strip()
    if chosen not in VARIANTS[key]:
        raise ValueError(f"Unknown {key} variant '{variant}'. Available: {sorted(VARIANTS[key])}")
    return key, chosen, CATALOG[VARIANTS[key][chosen]]


def list_variants(name):
    """Available variant names, including the default artifact; no downloads."""
    base, _, _ = _resolve(name)
    return sorted(VARIANTS[base])


def _modalities(spec):
    if spec.modalities is not None:
        return list(spec.modalities)
    if spec.category == "multimodal" or spec.name.startswith(("pascal", "coco_")) or spec.name in {
        "bdgp", "esp_game", "flickr30k", "iaprtc12", "nus_wide", "wikipedia",
    }:
        return ["image", "text"]
    if spec.name == "prokaryotic":
        return ["genomic", "text", "proteomic"]
    return ["features"] * spec.n_views


# ── Loading ────────────────────────────────────────────────────────

def _load(spec: DatasetSpec) -> MultiViewDataset:
    logger.info("Loading %s dataset...", spec.name)
    download_file(spec.url, spec.path)

    logger.debug("Reading dataset from: %s", spec.path)
    data = io.loadmat(spec.path)
    Xs, As, labels = spec.reader(data, **spec.kwargs)

    logger.info("Dataset loaded: samples=%d, views=%d", Xs[0].shape[0], len(Xs))
    return MultiViewDataset(Xs, As, labels, name=spec.name, modalities=_modalities(spec),
                            metadata={"source_url": spec.url, "artifact": spec.name})


def load_dataset(name: str, missing_rate: float = 0.0, *, variant=None, random_state=0) -> MultiViewDataset:
    """Load a dataset by name (case-insensitive).

    See :func:`list_datasets` for the available names.  Files are
    downloaded on first use and cached under ``~/.pymv_datasets``
    (override with the ``PYMV_DATA_DIR`` environment variable).
    """
    if not isinstance(missing_rate, Real) or not math.isfinite(missing_rate) or not 0 <= missing_rate < 1:
        raise ValueError("missing_rate must be in [0, 1)")
    base, chosen, spec = _resolve(name, variant)
    data = _load(spec)
    data.name = base
    data.metadata.update({"variant": chosen, "artifact": spec.name})
    if missing_rate:
        from .missing import simulate_missing_views
        data = simulate_missing_views(data, missing_rate, random_state=random_state)
    return data


load_data = load_dataset


def list_datasets(category: str | None = None, *, include_aliases=False) -> list[str]:
    """List available dataset names, optionally filtered by category.

    Parameters
    ----------
    category : str, optional
        One of ``"multiview"``, ``"multimodal"``,
        ``"attributed_graph_multiview"``, ``"attributed_graph_multimodal"``.
        Legacy broad filters ``"features"`` and ``"graph"`` are also accepted.
        A family matches when any variant belongs to that setting.
    include_aliases : bool, default=False
        Include deprecated artifact names for migration tooling.
    """
    candidates = CATALOG if include_aliases else VARIANTS
    names = [name for name in candidates if category is None or any(
        get_dataset_info(name, variant=v)["category"] == category
        or (category in {"features", "graph"} and CATALOG[artifact].category == category)
        for v, artifact in (VARIANTS[name].items() if name in VARIANTS else [(None, name)])
    )]
    return sorted(names)


def get_dataset_info(name: str, *, variant=None) -> dict:
    """Return metadata about a dataset without loading it.

    Keys: ``name``, ``category``, ``n_views``, ``description``,
    ``url``, ``cache_path``, ``cached``.
    """
    base, chosen, spec = _resolve(name, variant)
    modalities = _modalities(spec)
    has_graphs = spec.category in {"graph", "multimodal"}
    kind = "multimodal" if len(set(modalities)) > 1 else "multiview"
    return {
        "name": base,
        "variant": chosen,
        "variants": sorted(VARIANTS[base]),
        "artifact": spec.name,
        "category": f"attributed_graph_{kind}" if has_graphs else kind,
        "has_graphs": has_graphs,
        "modalities": modalities,
        "complete": True,
        "availability_note": "Hosted BBC4view artifact is known to be corrupt" if spec.name == "bbc4view" else "",
        "n_views": spec.n_views,
        "description": spec.description,
        "reference": spec.reference,
        "reference_verified": spec.reference_verified,
        "reference_details": spec.reference_details,
        "url": spec.url,
        "cache_path": str(spec.path),
        "cached": spec.path.exists(),
    }


# ── Per-dataset convenience loaders (load_acm, load_dblp, …) ──────

def _make_loader(spec: DatasetSpec) -> Callable[[], MultiViewDataset]:
    def loader(missing_rate=0.0, *, variant=None, random_state=0) -> MultiViewDataset:
        return load_dataset(spec.name, missing_rate, variant=variant, random_state=random_state)

    loader.__name__ = f"load_{spec.name}"
    loader.__qualname__ = loader.__name__
    loader.__doc__ = (
        f"Load the {spec.name} dataset — {spec.description or spec.category}.\n\n"
        f"Returns a :class:`~pymv.data.dataset.MultiViewDataset` with "
        f"{spec.n_views} views. Downloaded on first use and cached locally."
    )
    return loader


DATASETS: dict[str, Callable[[], MultiViewDataset]] = {
    name: _make_loader(spec) for name, spec in CATALOG.items()
}

for _name, _loader in DATASETS.items():
    globals()[f"load_{_name}"] = _loader


def _load_variant(base: str, variant: str | None) -> MultiViewDataset:
    if variant is None:
        return load_dataset(base)
    key = f"{base}_{variant.lower()}"
    if key not in CATALOG:
        available = sorted(
            n[len(base) + 1:] for n in CATALOG if n.startswith(f"{base}_")
        )
        raise ValueError(
            f"Unknown {base} variant '{variant}'. Available: {available}"
        )
    return load_dataset(key)


def load_pascal(variant: str | None = None, *, missing_rate=0.0, random_state=0) -> MultiViewDataset:
    """Load Pascal VOC. ``variant``: clip, resnet_minilm, vit_minilm, vit_sbert."""
    return load_dataset("pascal", missing_rate, variant=variant, random_state=random_state)


def load_coco_all(variant: str | None = None, *, missing_rate=0.0, random_state=0) -> MultiViewDataset:
    """Load MS-COCO (all). ``variant``: clip, resnet_minilm, vit_minilm, vit_sbert."""
    return load_dataset("coco_all", missing_rate, variant=variant, random_state=random_state)


def load_coco_cross(variant: str | None = None, *, missing_rate=0.0, random_state=0) -> MultiViewDataset:
    """Load MS-COCO (cross). ``variant``: clip, resnet_minilm, vit_minilm, vit_sbert."""
    return load_dataset("coco_cross", missing_rate, variant=variant, random_state=random_state)


__all__ = [
    "load_data",
    "list_variants",
    "VARIANTS",
    "DATASETS",
    "DatasetSpec",
    "CATALOG",
    "get_dataset_info",
    "list_datasets",
    "load_dataset",
    *sorted(f"load_{name}" for name in CATALOG),
]
