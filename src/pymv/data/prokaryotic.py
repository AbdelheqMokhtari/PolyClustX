import logging

import scipy.io as io

from .base import get_dataset_path
from .dataset import MultiViewDataset
from .downloader import download_file

logger = logging.getLogger(__name__)

PROKARYOTIC_URL = "https://zenodo.org/records/18923707/files/prokaryotic.mat?download=1"


def load_prokaryotic():

    logger.info("Loading Prokaryotic dataset...")

    dataset_dir = get_dataset_path("prokaryotic")

    file_path = dataset_dir / "Prokaryotic.mat"

    download_file(PROKARYOTIC_URL, file_path)

    logger.debug("Reading dataset from: %s", file_path)

    data = io.loadmat(file_path)

    X1 = data["gene_repert"]
    X2 = data["text"]
    X3 = data["proteome_comp"]

    Xs = [X1, X2, X3]

    labels = data["truth"].reshape(-1)

    As = [None] * len(Xs)

    logger.info("Dataset loaded: samples=%d, views=%d", Xs[0].shape[0], len(Xs))

    return MultiViewDataset(Xs, As, labels, name="prokaryotic")
