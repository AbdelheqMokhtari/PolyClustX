import logging

import scipy.io as io
import numpy as np

from .base import get_dataset_path
from .downloader import download_file

logger = logging.getLogger(__name__)

BDGP_URL = "https://zenodo.org/records/18923707/files/BDGP.mat?download=1" 


def load_bdgp():

    logger.info("Loading BDGP dataset...")

    dataset_dir = get_dataset_path("bdgp")

    file_path = dataset_dir / "BDGP.mat"

    download_file(BDGP_URL, file_path)

    logger.debug("Reading dataset from: %s", file_path)

    data = io.loadmat(file_path)

    X1 = data["X1"]
    X2 = data["X2"]

    Xs = [X1, X2]

    labels = data["Y"].reshape(-1)

    As = [None] * len(Xs)

    logger.info("Dataset loaded: samples=%d, views=%d", Xs[0].shape[0], len(Xs))

    return Xs, As, labels
