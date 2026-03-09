import scipy.io as io
import numpy as np

from .base import get_dataset_path
from .downloader import download_file


BDGP_URL = "https://zenodo.org/records/18923707/files/BDGP.mat?download=1" 


def load_bdgp():

    print("Loading BDGP dataset...")

    dataset_dir = get_dataset_path("bdgp")

    file_path = dataset_dir / "BDGP.mat"

    download_file(BDGP_URL, file_path)

    print(f"Reading dataset from: {file_path}")

    data = io.loadmat(file_path)

    X1 = data["X1"]
    X2 = data["X2"]

    Xs = [X1, X2]

    labels = data["Y"].reshape(-1)

    As = [None] * len(Xs)

    print("Dataset loaded successfully.")
    print(f"Samples: {Xs[0].shape[0]}")
    print(f"Views: {len(Xs)}")

    return Xs, As, labels
