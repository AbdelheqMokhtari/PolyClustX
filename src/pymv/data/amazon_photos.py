import logging

import scipy.io as io

from .base import get_dataset_path
from .dataset import MultiViewDataset
from .downloader import download_file

logger = logging.getLogger(__name__)

AMAZON_URL = "https://zenodo.org/records/18923707/files/Amazon_photos.mat?download=1"


def load_amazon_photos():

    logger.info("Loading Amazon photos dataset...")

    dataset_dir = get_dataset_path("amazon_photos")

    file_path = dataset_dir / "amazon_photos.mat"

    download_file(AMAZON_URL, file_path)

    logger.debug("Reading dataset from: %s", file_path)

    data = io.loadmat(file_path)

    X = data['features']
    A = data.get('adj')
    labels = data['label']
    labels = labels.reshape(-1)

    Xs = [X, X @ X.T]

    As = [A.toarray() for _ in Xs]

    logger.info("Dataset loaded: samples=%d, views=%d", Xs[0].shape[0], len(Xs))

    return MultiViewDataset(Xs, As, labels, name="amazon_photos")



