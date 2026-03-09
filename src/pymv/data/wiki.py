import scipy.io as io
import numpy as np
from sklearn.neighbors import kneighbors_graph

from .base import get_dataset_path
from .downloader import download_file


WIKI_URL = "https://zenodo.org/records/18923707/files/wiki.mat?download=1"


def load_wiki():

    print("Loading Wiki dataset...")

    dataset_dir = get_dataset_path("wiki")

    file_path = dataset_dir / "wiki.mat"

    download_file(WIKI_URL, file_path)

    print(f"Reading dataset from: {file_path}")

    data = io.loadmat(file_path)

    X = data["fea"].toarray().astype(float)
    A = data["W"]

    labels = data["gnd"].reshape(-1)

    Xs = [X, np.log2(1 + X)]
    As = [A, kneighbors_graph(X, 5, metric='cosine')]

    print("Dataset loaded successfully.")
    print(f"Samples: {Xs[0].shape[0]}")
    print(f"Views: {len(Xs)}")

    return Xs, As, labels
