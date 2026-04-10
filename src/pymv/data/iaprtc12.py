import logging

import scipy.io as io

from .base import get_dataset_path
from .dataset import MultiViewDataset
from .downloader import download_file

logger = logging.getLogger(__name__)

IAPRTC12_URL = "https://zenodo.org/records/19470217/files/iaprtc12.mat?download=1"


def load_iaprtc12():

    logger.info("Loading IAPR-TC12 dataset...")

    dataset_dir = get_dataset_path("iaprtc12")

    file_path = dataset_dir / "iaprtc12.mat"

    download_file(IAPRTC12_URL, file_path)

    logger.debug("Reading dataset from: %s", file_path)

    data = io.loadmat(file_path)

    X_img = data["img"]
    X_txt = data["txt"]

    Xs = [X_img, X_txt]

    labels = data["label"].reshape(-1)

    As = [None] * len(Xs)

    logger.info("Dataset loaded: samples=%d, views=%d", Xs[0].shape[0], len(Xs))

    return MultiViewDataset(Xs, As, labels, name="iaprtc12")
