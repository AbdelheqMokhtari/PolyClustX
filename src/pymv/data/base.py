from pathlib import Path

DATA_DIR = Path.home() / ".pymv_datasets"


def get_dataset_path(name):

    path = DATA_DIR / name

    path.mkdir(parents=True, exist_ok=True)

    return path