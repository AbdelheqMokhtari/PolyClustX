import logging

import scipy.io as io

from .base import get_dataset_path
from .dataset import MultiViewDataset
from .downloader import download_file

logger = logging.getLogger(__name__)

IMDB_URL = "https://zenodo.org/records/18923707/files/IMDB.mat?download=1"


def load_imdb():

    logger.info("Loading IMDB dataset...")

    dataset_dir = get_dataset_path("imdb")

    file_path = dataset_dir / "IMDB.mat"

    download_file(IMDB_URL, file_path)

    logger.debug("Reading dataset from: %s", file_path)

    data = io.loadmat(file_path)

    X = data["features"]
    A = data["MAM"]
    B = data["MDM"]

    labels = data["label"].reshape(-1)

    As = [A.toarray(), B.toarray()]
    Xs = [X.toarray() for _ in As]

    logger.info("Dataset loaded: samples=%d, views=%d", Xs[0].shape[0], len(Xs))

    return MultiViewDataset(Xs, As, labels, name="imdb")
