import logging
from functools import partial

import scipy.io as io

from .base import get_dataset_path
from .dataset import MultiViewDataset
from .downloader import download_file

logger = logging.getLogger(__name__)

PASCAL_URL = "https://zenodo.org/records/18923707/files/pascal.mat?download=1"

_PASCAL_VARIANTS = {
    "clip": "https://zenodo.org/records/19470217/files/pascal_clip.mat?download=1",
    "resnet_minilm": "https://zenodo.org/records/19470217/files/pascal_resnet_minilm.mat?download=1",
    "vit_minilm": "https://zenodo.org/records/19470217/files/pascal_vit_minilm.mat?download=1",
    "vit_sbert": "https://zenodo.org/records/19470217/files/pascal_vit_sbert.mat?download=1",
}


def load_pascal(variant=None):

    if variant is not None:
        variant = variant.lower()
        if variant not in _PASCAL_VARIANTS:
            raise ValueError(
                f"Unknown Pascal variant '{variant}'. "
                f"Available: {list(_PASCAL_VARIANTS.keys())}"
            )
        url = _PASCAL_VARIANTS[variant]
        filename = f"pascal_{variant}.mat"
        name = f"pascal_{variant}"
        logger.info("Loading Pascal (%s) dataset...", variant)
    else:
        url = PASCAL_URL
        filename = "pascal.mat"
        name = "pascal"
        logger.info("Loading Pascal dataset...")

    dataset_dir = get_dataset_path("pascal")

    file_path = dataset_dir / filename

    download_file(url, file_path)

    logger.debug("Reading dataset from: %s", file_path)

    data = io.loadmat(file_path)

    X1 = data["X1"]
    X2 = data["X2"]

    Xs = [X1, X2]

    labels = data["Y"].reshape(-1)

    As = [None] * len(Xs)

    logger.info("Dataset loaded: samples=%d, views=%d", Xs[0].shape[0], len(Xs))

    return MultiViewDataset(Xs, As, labels, name=name)


load_pascal_clip = partial(load_pascal, variant="clip")
load_pascal_resnet_minilm = partial(load_pascal, variant="resnet_minilm")
load_pascal_vit_minilm = partial(load_pascal, variant="vit_minilm")
load_pascal_vit_sbert = partial(load_pascal, variant="vit_sbert")
