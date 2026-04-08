import logging

import numpy as np
import scipy.io as io
from sklearn.neighbors import kneighbors_graph

from .base import get_dataset_path
from .dataset import MultiViewDataset
from .downloader import download_file

logger = logging.getLogger(__name__)

WIKI_URL = "https://zenodo.org/records/18923707/files/wiki.mat?download=1"


def load_wiki():

    logger.info("Loading Wiki dataset...")

    dataset_dir = get_dataset_path("wiki")

    file_path = dataset_dir / "wiki.mat"

    download_file(WIKI_URL, file_path)

    logger.debug("Reading dataset from: %s", file_path)

    data = io.loadmat(file_path)

    X = data["fea"].toarray().astype(float)
    A = data["W"]

    labels = data["gnd"].reshape(-1)

    Xs = [X, np.log2(1 + X)]
    As = [A, kneighbors_graph(X, 5, metric='cosine')]

    logger.info("Dataset loaded: samples=%d, views=%d", Xs[0].shape[0], len(Xs))

    return MultiViewDataset(Xs, As, labels, name="wiki")
