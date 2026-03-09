import scipy.io as io

from .base import get_dataset_path
from .downloader import download_file


AMAZON_URL = "https://zenodo.org/records/18923707/files/Amazon_photos.mat?download=1"


def load_amazon_photos():

    print("Loading Amazon photos dataset...")

    dataset_dir = get_dataset_path("amazon_photos")

    file_path = dataset_dir / "amazon_photos.mat"

    download_file(AMAZON_URL, file_path)

    print(f"Reading dataset from: {file_path}")

    data = io.loadmat(file_path)

    X = data['features']
    A = data.get('adj')
    labels = data['label']
    labels = labels.reshape(-1)

    Xs = [X, X @ X.T]

    As = [A.toarray() for _ in Xs]

    print("Dataset loaded successfully.")
    print(f"Samples: {Xs[0].shape[0]}")
    print(f"Views: {len(Xs)}")

    return Xs, As, labels



