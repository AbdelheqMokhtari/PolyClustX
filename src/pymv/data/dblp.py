import scipy.io as io

from .base import get_dataset_path
from .downloader import download_file


DBLP_URL = "https://zenodo.org/records/18923707/files/DBLP.mat?download=1"


def load_dblp():

    print("Loading DBLP dataset...")

    dataset_dir = get_dataset_path("dblp")

    file_path = dataset_dir / "DBLP.mat"

    download_file(DBLP_URL, file_path)

    print(f"Reading dataset from: {file_path}")

    data = io.loadmat(file_path)

    X = data['features']
    A = data['net_APTPA']
    B = data['net_APCPA']
    C = data['net_APA']

    labels = data["label"].reshape(-1)

    As = [A.toarray(), B.toarray(), C.toarray()]
    Xs = [X.toarray() for _ in As]

    print("Dataset loaded successfully.")
    print(f"Samples: {Xs[0].shape[0]}")
    print(f"Views: {len(Xs)}")

    return Xs, As, labels
