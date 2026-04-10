import logging
from functools import partial

import scipy.io as io

from .base import get_dataset_path
from .dataset import MultiViewDataset
from .downloader import download_file

logger = logging.getLogger(__name__)

COCO_CROSS_URL = "https://zenodo.org/records/18923707/files/coco_cross.mat?download=1"

_COCO_CROSS_VARIANTS = {
    "clip": "https://zenodo.org/records/19470217/files/coco_cross_clip.mat?download=1",
    "resnet_minilm": "https://zenodo.org/records/19470217/files/coco_cross_resnet_minilm.mat?download=1",
    "vit_minilm": "https://zenodo.org/records/19470217/files/coco_cross_vit_minilm.mat?download=1",
    "vit_sbert": "https://zenodo.org/records/19470217/files/coco_cross_vit_sbert.mat?download=1",
}


def load_coco_cross(variant=None):

    if variant is not None:
        variant = variant.lower()
        if variant not in _COCO_CROSS_VARIANTS:
            raise ValueError(
                f"Unknown COCO (cross) variant '{variant}'. "
                f"Available: {list(_COCO_CROSS_VARIANTS.keys())}"
            )
        url = _COCO_CROSS_VARIANTS[variant]
        filename = f"coco_cross_{variant}.mat"
        name = f"coco_cross_{variant}"
        logger.info("Loading COCO (cross, %s) dataset...", variant)
    else:
        url = COCO_CROSS_URL
        filename = "coco_cross.mat"
        name = "coco_cross"
        logger.info("Loading COCO (cross) dataset...")

    dataset_dir = get_dataset_path("coco_cross")

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


load_coco_cross_clip = partial(load_coco_cross, variant="clip")
load_coco_cross_resnet_minilm = partial(load_coco_cross, variant="resnet_minilm")
load_coco_cross_vit_minilm = partial(load_coco_cross, variant="vit_minilm")
load_coco_cross_vit_sbert = partial(load_coco_cross, variant="vit_sbert")
