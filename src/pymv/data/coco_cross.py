import scipy.io as io
import numpy as np

from .base import get_dataset_path
from .downloader import download_file


COCO_CROSS_URL = "https://zenodo.org/records/18923707/files/coco_cross.mat?download=1" 


def load_coco_cross():

    print("Loading COCO (cross) dataset...")

    dataset_dir = get_dataset_path("coco_cross")

    file_path = dataset_dir / "coco_cross.mat"

    download_file(COCO_CROSS_URL, file_path)

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
