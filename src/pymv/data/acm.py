import logging

import scipy.io as io

from .base import get_dataset_path
from .dataset import MultiViewDataset
from .downloader import download_file

logger = logging.getLogger(__name__)

ACM_URL = "https://zenodo.org/records/18923707/files/ACM.mat?download=1"


def load_acm():

    logger.info("Loading ACM dataset...")

    dataset_dir = get_dataset_path("acm")

    file_path = dataset_dir / "ACM.mat"

    download_file(ACM_URL, file_path)

    logger.debug("Reading dataset from: %s", file_path)

    data = io.loadmat(file_path)

    X = data["features"]
    A = data["PAP"]
    B = data["PLP"]

    labels = data["label"].reshape(-1)

    As = [A.toarray(), B.toarray()]
    Xs = [X.toarray() for _ in As]

    logger.info("Dataset loaded: samples=%d, views=%d", Xs[0].shape[0], len(Xs))

    return MultiViewDataset(Xs, As, labels, name="acm")
