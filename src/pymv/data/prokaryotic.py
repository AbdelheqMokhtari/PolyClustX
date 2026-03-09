import scipy.io as io
import numpy as np

from .base import get_dataset_path
from .downloader import download_file


PROKARYOTIC_URL = "https://zenodo.org/records/18923707/files/prokaryotic.mat?download=1"


def load_prokaryotic():

    print("Loading Prokaryotic dataset...")

    dataset_dir = get_dataset_path("prokaryotic")

    file_path = dataset_dir / "Prokaryotic.mat"

    download_file(PROKARYOTIC_URL, file_path)

    print(f"Reading dataset from: {file_path}")

    data = io.loadmat(file_path)

    X1 = data["gene_repert"]
    X2 = data["text"]
    X3 = data["proteome_comp"]

    Xs = [X1, X2, X3]

    labels = data["truth"].reshape(-1)

    As = [None] * len(Xs)

    print("Dataset loaded successfully.")
    print(f"Samples: {Xs[0].shape[0]}")
    print(f"Views: {len(Xs)}")

    return Xs, As, labels
