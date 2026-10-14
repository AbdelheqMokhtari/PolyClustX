"""Dataset families, representation variants and direct Hugging Face downloads.

Every catalog artifact uses X* features, A* adjacencies and labels. Adding
a DatasetSpec automatically exposes it to Python, the CLI and the interface.
"""
from __future__ import annotations

import difflib
import logging
import math
import warnings
from dataclasses import dataclass, field
from numbers import Real
from typing import Callable

from scipy.io import whosmat

from .base import get_data_home
from .dataset import MultiViewDataset
from .downloader import download_file
from .importers import _selection, array_roles, dataset_from_arrays, default_views, read_arrays

logger = logging.getLogger(__name__)
_DOWNLOAD_ROOT = "https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main"
_REFERENCE_COLLECTION = "https://github.com/ChuanbinZhang/Multi-view-datasets"
_GRAPH_REFERENCE = "https://github.com/chakib401/LMGEC"
_MULTIMODAL_GRAPH_REFERENCE = "https://github.com/sktsherlock/MAGB"
CATEGORY_LABELS = {
    "attributed_graph_multiview": "Multi-view attributed graph clustering (MVAGC)",
    "multiview": "Multi-view clustering (MVC)",
    "multimodal": "Multimodal clustering (MMC)",
    "attributed_graph_multimodal": "Multimodal attributed graph clustering (MMAGC)",
}


@dataclass(frozen=True)
class DatasetSpec:
    """One downloadable artifact; a family can group several encoder variants."""

    name: str
    url: str
    filename: str
    reader: Callable | None = None
    kwargs: dict = field(default_factory=dict)
    category: str = "multiview"
    n_views: int = 2
    description: str = ""
    cache_dir: str | None = None
    reference: str = _REFERENCE_COLLECTION
    reference_verified: bool = False
    reference_details: str = ""
    modalities: tuple[str, ...] | None = None
    n_samples: int | None = None
    n_clusters: int | None = None
    feature_dims: tuple[int, ...] = ()
    feature_descriptions: tuple[str, ...] = ()
    graph_descriptions: tuple[str, ...] = ()
    encoders: str = ""
    graph_statistics: dict = field(default_factory=dict)

    @property
    def path(self):
        # A separate source namespace never reuses a differently structured
        # old Zenodo artifact. Metadata lookups do not create cache folders.
        return get_data_home() / "huggingface" / (self.cache_dir or self.name) / self.filename


def _spec(name, folder, samples, classes, features, description, *,
          graphs=(), modalities=None, dimensions=(), filename=None, encoders="", statistics=None):
    filename = filename or f"{folder}.mat"
    kind = "multimodal" if modalities and len(set(modalities)) > 1 else "multiview"
    category = f"attributed_graph_{kind}" if graphs else kind
    return DatasetSpec(
        name=name, url=f"{_DOWNLOAD_ROOT}/{folder}/{filename}?download=true",
        filename=filename, cache_dir=folder, category=category,
        n_views=len(graphs) if len(features) == 1 and graphs else len(features),
        n_samples=samples, n_clusters=classes, feature_dims=tuple(dimensions),
        description=description, feature_descriptions=tuple(features), graph_descriptions=tuple(graphs),
        modalities=tuple(modalities or ["features"] * len(features)), encoders=encoders,
        graph_statistics=statistics or {},
        reference=(_MULTIMODAL_GRAPH_REFERENCE if graphs and kind == "multimodal"
                   else _GRAPH_REFERENCE if graphs else _REFERENCE_COLLECTION),
        reference_verified=not graphs and kind == "multiview",
    )


def _descriptors(number):
    return tuple(f"Visual descriptor {index}" for index in range(1, number + 1))


def _image_text(name, folder, samples, classes, description):
    common = dict(name=name, folder=folder, samples=samples, classes=classes,
                  features=("Image embedding", "Caption embedding"), description=description,
                  modalities=("image", "text"))
    return [
        _spec(**common, dimensions=(2048, 300), encoders="ResNet + Doc2Vec"),
        *[_spec(**{**common, "name": f"{name}_{variant}"}, filename=f"{folder}_{suffix}.mat",
                dimensions=dimensions, encoders=encoders)
          for variant, suffix, dimensions, encoders in (
              ("clip", "Clip", (512, 512), "CLIP + CLIP"),
              ("resnet_minilm", "resnet_minilm", (2048, 384), "ResNet + MiniLM"),
              ("vit_minilm", "vit_minilm", (768, 384), "ViT + MiniLM"),
              ("vit_sbert", "vit_sbert", (768, 768), "ViT + SBERT"),
          )],
    ]


def _multimodal_graph(name, folder, samples, classes, description, arcs, pairs, loops, symmetric, extra=False):
    common = dict(name=name, folder=folder, samples=samples, classes=classes, description=description,
                  features=("Product/post image embedding", "Description/caption embedding"),
                  modalities=("image", "text"),
                  graphs=("Purchasing/viewing links" if name in {"toys", "movies", "grocery", "grocery_s"}
                          else "Posts with a shared commenting user" if name in {"reddit", "reddit_s"}
                          else "Provided relationships; exact semantics are undocumented",),
                  statistics={"stored_arcs": arcs, "unordered_pairs": pairs,
                              "self_loops": loops, "symmetric": symmetric})
    variants = [("clip", "Clip", (768, 768), "CLIP image + CLIP text")]
    if extra:
        variants += [
            ("clip_minilm", "Clip_minilm", (512, 384), "CLIP + MiniLM"),
            ("resnet_clip", "resnet_Clip", (2048, 512), "ResNet + CLIP"),
            ("resnet_minilm", "resnet_minilm", (2048, 384), "ResNet + MiniLM"),
        ]
    return [
        _spec(**common, dimensions=(768, 3072), encoders="CLIP-ViT + Gemma-7B"),
        *[_spec(**{**common, "name": f"{name}_{variant}"}, filename=f"{folder}_{suffix}.mat",
                dimensions=dimensions, encoders=encoders)
          for variant, suffix, dimensions, encoders in variants],
    ]


_SPECS = [
    _spec("acm", "ACM", 3025, 3, ("Keyword-based paper features",),
          "Academic papers in database, wireless communication and data mining topics.",
          dimensions=(1870,), graphs=("PAP: papers sharing authors", "PLP: papers sharing subjects")),
    _spec("dblp", "DBLP", 4057, 4, ("Publication keyword features of authors",),
          "Authors grouped into four research areas.", dimensions=(334,),
          graphs=("APTPA: authors connected through paper terms", "APCPA: authors sharing venues",
                  "APA: co-authorship")),
    _spec("imdb", "IMDB", 4780, 3, ("Movie plot text features",),
          "Movies grouped into action, comedy and drama.", dimensions=(1232,),
          graphs=("MAM: movies sharing actors", "MDM: movies sharing directors")),
    _spec("wiki", "Wiki", 2405, 17, ("Webpage text attributes",),
          "Webpages in seventeen topics with a hyperlink graph.", dimensions=(4973,),
          graphs=("W: hyperlinks between webpages",)),
    _spec("amazon_photos", "Amazon_photos", 7487, 8, ("Product attributes",),
          "Photography products in an Amazon co-purchasing network.", dimensions=(745,),
          graphs=("Product co-purchasing links",)),
    _spec("100leaves", "100leaves", 1600, 100, ("Shape", "Boundary", "Texture"),
          "Plant leaves from one hundred species.", dimensions=(64, 64, 64)),
    _spec("3sources", "3sources", 169, 6, ("BBC text", "Guardian text", "Reuters text"),
          "News stories covered by three publishers.", dimensions=(3560, 3631, 3068)),
    _spec("aloi", "aloi", 10800, 100, ("Color statistics", "Haralick texture", "HSV histogram", "RGB histogram"),
          "Object photographs under varying lighting and rotation.", dimensions=(77, 13, 64, 125)),
    _spec("bbcsport", "bbcsport", 544, 5, ("Text representation 1", "Text representation 2"),
          "BBC sports articles represented by two text views.", dimensions=(3183, 3203)),
    _spec("caltech101_7", "caltech101-7", 1474, 7, ("Gabor", "Wavelet moments", "CENT", "HOG", "GIST", "LBP"),
          "Seven-category object-image subset.", dimensions=(48, 40, 254, 1984, 512, 928)),
    _spec("handwritten", "handwritten", 2000, 10,
          ("Profile correlations", "Fourier coefficients", "Karhunen–Loève coefficients",
           "Morphological features", "Pixel averages", "Zernike moments"),
          "Handwritten digits represented by six numerical feature sets.", dimensions=(216, 76, 64, 6, 240, 47)),
    _spec("mnist_4", "mnist-4", 4000, 4, _descriptors(3),
          "MNIST digits 0–3 with three precomputed feature representations."),
    _spec("mnist_10k", "mnist-10k", 10000, 10, _descriptors(3),
          "MNIST digits 0–9 with three precomputed feature representations."),
    _spec("msrc_v5", "msrc-v5", 210, 7, _descriptors(5),
          "Object and scene images with five visual feature sets.", dimensions=(24, 576, 512, 256, 254)),
    _spec("outdoor_scene", "outdoorscence", 2688, 8,
          ("Global appearance", "Gradient descriptor", "Texture descriptor", "Visual descriptor 4"),
          "Outdoor scenes represented by four visual feature sets."),
    _spec("protein_fold", "proteinfold", 694, 27,
          tuple(f"Numerical protein descriptor {i}" for i in range(1, 13)),
          "Protein domains grouped by structural fold.", dimensions=(27,) * 12),
    _spec("reuters_1200", "reuters-1200", 1200, 6,
          tuple(f"Language-specific text representation {i}" for i in range(1, 6)),
          "News documents in five languages with reduced vocabularies.", dimensions=(2000,) * 5),
    _spec("reuters_1500", "reuters-1500", 1500, 6,
          tuple(f"Language-specific text representation {i}" for i in range(1, 6)),
          "Multilingual news documents with larger language-specific vocabularies."),
    _spec("uci", "uci", 2000, 10, ("Pixel features", "Fourier features", "Morphological features"),
          "Handwritten digits with three selected feature sets.", dimensions=(240, 76, 6)),
    _spec("webkb", "webkb", 1051, 2, ("Page content", "Hyperlink anchor text"),
          "University webpages represented by two text views."),
    _spec("wikipedia_test", "wikipedia-test", 693, 10, ("Article text", "Image features"),
          "Wikipedia test subset pairing article text with images.", modalities=("text", "image")),
    _spec("yale", "yale", 165, 15, ("Intensity", "Texture", "Gabor"),
          "Face images of fifteen people.", dimensions=(4096, 3304, 6750)),
    *_image_text("pascal", "pascal", 1000, 20,
                 "Pascal Sentences: images with human-written captions; fifty images per category."),
    *_image_text("coco_cross", "coco-cross", 7429, 10,
                 "COCO image-caption subset selecting a category from each of ten supercategories."),
    *_image_text("coco_all", "coco-all", 23189, 43,
                 "Broader COCO image-caption subset retaining sufficiently populated categories."),
    *[_spec(name, folder, samples, classes, ("VGG-16 image features", "BERT text features"), description,
            dimensions=(4096, 768), modalities=("image", "text"), encoders="VGG-16 + BERT")
      for name, folder, samples, classes, description in (
          ("esp_game", "esp-game", 11032, 7, "Images and descriptive tags from an online labeling game."),
          ("iapr_tc12", "iapr-tc12", 7855, 6, "Photographs with textual annotations for cross-language retrieval."),
          ("mirflickr", "mirflickr", 12154, 6, "Flickr photographs with user tags and semantic annotations."),
          ("nus_wide", "nus-wide", 20000, 8, "Processed subset of a Flickr image/tag benchmark."),
      )],
    *_multimodal_graph("toys", "toys", 20695, 18, "Amazon Toys and Games products.",
                      63443, 56701, 22, False, extra=True),
    *_multimodal_graph("movies", "movies", 16672, 20, "Amazon Movies and TV products.",
                      109195, 80401, 0, False, extra=True),
    *_multimodal_graph("grocery_s", "grocery-S", 17074, 20,
                      "Smaller Amazon grocery benchmark; subset construction is undocumented.",
                      85670, 71131, 1, False, extra=True),
    *_multimodal_graph("grocery", "grocery", 84379, 20, "Amazon grocery and gourmet food products.",
                      962957, 795041, 18, False),
    *_multimodal_graph("reddit_s", "reddit-s", 15894, 20, "Smaller Reddit image-caption graph.",
                      283080, 141540, 0, True),
    *_multimodal_graph("reddit", "reddit", 99638, 50, "Larger Reddit image-caption graph.",
                      583594, 291797, 0, True),
    *_multimodal_graph("photo", "photo", 36731, 12,
                      "Image/text node features with a graph; exact domain and relationship meanings are undocumented.",
                      349707, 307979, 8, False),
    *_multimodal_graph("arts", "Arts", 58487, 11,
                      "Image/text node features with a graph; exact origin and category meanings are undocumented.",
                      472462, 406126, 6, False),
]
CATALOG = {spec.name: spec for spec in _SPECS}
VARIANTS = {name: {"default": name} for name in CATALOG}
for _base in ("pascal", "coco_all", "coco_cross", "arts", "grocery", "grocery_s", "movies",
              "photo", "reddit", "reddit_s", "toys"):
    for _variant in ("clip", "resnet_minilm", "vit_minilm", "vit_sbert", "clip_minilm", "resnet_clip"):
        _artifact = f"{_base}_{_variant}"
        if _artifact in CATALOG:
            VARIANTS[_base][_variant] = _artifact
            VARIANTS.pop(_artifact)
ALIASES = {artifact: (base, variant) for base, variants in VARIANTS.items()
           for variant, artifact in variants.items() if artifact != base}
ALIASES.update({
    "arts_multimodal": ("arts", "default"), "grocery_multimodal": ("grocery", "default"),
    "grocerys_multimodal": ("grocery_s", "default"), "movies_multimodal": ("movies", "default"),
    "photo_multimodal": ("photo", "default"), "reddit_multimodal": ("reddit", "default"),
    "reddits_multimodal": ("reddit_s", "default"), "toys_multimodal": ("toys", "default"),
    "iaprtc12": ("iapr_tc12", "default"), "wikipedia": ("wikipedia_test", "default"),
    "outdoorscence": ("outdoor_scene", "default"), "proteinfold": ("protein_fold", "default"),
})


def _resolve(name, variant=None):
    key = str(name).lower().strip().replace("-", "_")
    if key in ALIASES:
        base, chosen = ALIASES[key]
        if chosen != "default" and variant not in (None, "default"):
            raise ValueError("Do not combine a variant artifact name with variant=")
        warnings.warn(f"Use load_data({base!r}, variant={chosen!r}) instead of {key!r}",
                      DeprecationWarning, stacklevel=3)
        key, variant = base, chosen if chosen != "default" else variant
    if key not in VARIANTS:
        close = difflib.get_close_matches(key, VARIANTS, n=3, cutoff=0.6)
        raise ValueError(f"Unknown dataset '{name}'. Did you mean {close}? Available: {sorted(VARIANTS)}")
    chosen = "default" if variant is None else str(variant).lower().strip().replace("-", "_")
    if chosen not in VARIANTS[key]:
        raise ValueError(f"Unknown {key} variant '{variant}'. Available: {sorted(VARIANTS[key])}")
    return key, chosen, CATALOG[VARIANTS[key][chosen]]


def list_variants(name):
    """Available encoder variants, including default; never downloads data."""
    base, _, _ = _resolve(name)
    return sorted(VARIANTS[base])


def _load(spec, views=None):
    download_file(spec.url, spec.path)
    if spec.reader is not None:
        arrays = read_arrays(spec.path)
        Xs, As, labels = spec.reader(arrays, **spec.kwargs)
        data = MultiViewDataset(Xs, As, labels, modalities=spec.modalities)
        if views is not None:
            raise ValueError("Custom readers must handle view selection themselves")
    else:
        try:
            headers = {key: {"shape": list(shape), "dtype": dtype}
                       for key, shape, dtype in whosmat(spec.path)}
        except NotImplementedError as exc:
            raise ValueError("MATLAB v7.3 is unsupported; export MATLAB v7 or NPZ") from exc
        if "labels" not in headers:
            raise ValueError(f"{spec.filename} must contain a 'labels' vector")
        chosen = default_views(headers) if views is None else views
        feature_keys, relations, _ = _selection(headers, chosen)
        selected = {"labels", *feature_keys, *(key for keys in relations for key in keys)}
        arrays = read_arrays(spec.path, variable_names=sorted(selected))
        data = dataset_from_arrays(arrays, views=chosen, label_key="labels")
        available = array_roles(headers)
        features, graphs = available["features"], available["graphs"]
        data.metadata.update(source_arrays=headers, available_features=features, available_graphs=graphs)
        selection = list(data.metadata["view_selection"].values())
        modalities = spec.modalities or ("features",) * len(features)
        data.modalities = [modalities[features.index(feature)] if feature in features and
                           features.index(feature) < len(modalities) else "features" for feature, _ in selection]
        data.metadata["feature_descriptions"] = {
            key: spec.feature_descriptions[index] if index < len(spec.feature_descriptions) else key
            for index, key in enumerate(features)
        }
        data.metadata["graph_descriptions"] = {
            key: next((description for description in spec.graph_descriptions
                       if description.split(":", 1)[0].lower() == key.rsplit("_", 1)[-1].lower()),
                      spec.graph_descriptions[index] if index < len(spec.graph_descriptions) else key)
            for index, key in enumerate(graphs)
        }
        data.metadata["feature_modalities"] = dict(zip(features, modalities))
    data.name = spec.name
    data.metadata.update(source_url=spec.url, artifact=spec.name, description=spec.description,
                         reference=spec.reference, encoders=spec.encoders)
    return data


def load_dataset(name, missing_rate=0.0, *, variant=None, random_state=0, views=None):
    """Load a family/encoder variant, optionally choosing named feature/graph views.

    For example, views={"coauthor": ("X1_features", "A1_PAP"),
    "fused": ("X1_features", ["A1_PAP", "A2_PLP"])}. Omit graphs with
    views={"keywords": "X1_features"}. Labels are used only for evaluation.
    """
    if not isinstance(missing_rate, Real) or not math.isfinite(missing_rate) or not 0 <= missing_rate < 1:
        raise ValueError("missing_rate must be in [0, 1)")
    base, chosen, spec = _resolve(name, variant)
    data = _load(spec) if views is None else _load(spec, views=views)
    data.name = base
    data.metadata.update(dataset=base, variant=chosen, artifact=spec.name)
    if missing_rate:
        from .missing import simulate_missing_views
        data = simulate_missing_views(data, missing_rate, random_state=random_state)
    return data


load_data = load_dataset


def get_dataset_info(name, *, variant=None):
    """Describe an artifact, encoders, feature/graph semantics and expected sizes."""
    base, chosen, spec = _resolve(name, variant)
    has_graphs = spec.category.startswith("attributed_graph_")
    modalities = list(spec.modalities or ["features"] * spec.n_views)
    return {
        "name": base, "variant": chosen, "variants": sorted(VARIANTS[base]), "artifact": spec.name,
        "category": spec.category, "category_label": CATEGORY_LABELS.get(spec.category, spec.category),
        "has_graphs": has_graphs, "modalities": modalities, "complete": True, "availability_note": "",
        "n_views": spec.n_views, "n_feature_views": len(spec.feature_descriptions) or len(modalities),
        "n_graphs": len(spec.graph_descriptions), "n_samples": spec.n_samples, "n_clusters": spec.n_clusters,
        "feature_dims": list(spec.feature_dims), "feature_descriptions": list(spec.feature_descriptions),
        "graph_descriptions": list(spec.graph_descriptions), "graph_statistics": dict(spec.graph_statistics),
        "encoders": spec.encoders, "description": spec.description, "reference": spec.reference,
        "reference_verified": spec.reference_verified, "reference_details": spec.reference_details,
        "url": spec.url, "cache_path": str(spec.path), "cached": spec.path.exists(),
    }


def list_datasets(category=None, *, include_aliases=False):
    """List families; filter by the four settings or broad features/graph groups."""
    candidates = set(VARIANTS) | (set(CATALOG) | set(ALIASES) if include_aliases else set())
    def matches(name):
        base, _, _ = _resolve(name)
        specs = [CATALOG[artifact] for artifact in VARIANTS[base].values()]
        return category is None or any(
            spec.category == category
            or category == "graph" and spec.category.startswith("attributed_graph_")
            or category == "features" and not spec.category.startswith("attributed_graph_")
            for spec in specs
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return sorted(name for name in candidates if matches(name))


def _make_loader(name):
    def loader(missing_rate=0.0, *, variant=None, random_state=0, views=None):
        return load_dataset(name, missing_rate, variant=variant, random_state=random_state, views=views)
    loader.__name__ = f"load_{name}"
    loader.__qualname__ = loader.__name__
    loader.__doc__ = f"Load {name}; accepts variant, missing_rate and a named views mapping."
    return loader


DATASETS = {name: _make_loader(name) for name in set(CATALOG) | set(ALIASES)}
for _name, _loader in DATASETS.items():
    globals()[f"load_{_name}"] = _loader


def load_pascal(variant=None, *, missing_rate=0.0, random_state=0, views=None):
    """Load Pascal Sentences with an optional encoder variant."""
    return load_dataset("pascal", missing_rate, variant=variant, random_state=random_state, views=views)


def load_coco_all(variant=None, *, missing_rate=0.0, random_state=0, views=None):
    """Load COCO-all with an optional encoder variant."""
    return load_dataset("coco_all", missing_rate, variant=variant, random_state=random_state, views=views)


def load_coco_cross(variant=None, *, missing_rate=0.0, random_state=0, views=None):
    """Load COCO-cross with an optional encoder variant."""
    return load_dataset("coco_cross", missing_rate, variant=variant, random_state=random_state, views=views)


__all__ = ["load_data", "list_variants", "VARIANTS", "ALIASES", "DATASETS", "DatasetSpec", "CATALOG",
           "CATEGORY_LABELS", "get_dataset_info", "list_datasets", "load_dataset",
           *sorted(f"load_{name}" for name in DATASETS)]
