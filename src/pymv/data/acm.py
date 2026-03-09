import scipy.io as io

from .base import get_dataset_path
from .downloader import download_file


ACM_URL = "https://zenodo.org/records/18922905/files/ACM.mat?download=1"


def load_acm():

    print("Loading ACM dataset...")

    dataset_dir = get_dataset_path("acm")

    file_path = dataset_dir / "ACM.mat"

    download_file(ACM_URL, file_path)

    print(f"Reading dataset from: {file_path}")

    data = io.loadmat(file_path)

    X = data["features"]
    A = data["PAP"]
    B = data["PLP"]

    labels = data["label"].reshape(-1)

    As = [A.toarray(), B.toarray()]
    Xs = [X.toarray() for _ in As]

    print("Dataset loaded successfully.")
    print(f"Samples: {Xs[0].shape[0]}")
    print(f"Views: {len(Xs)}")

    return Xs, As, labels



