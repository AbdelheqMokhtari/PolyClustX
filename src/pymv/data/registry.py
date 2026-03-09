from .acm import load_acm


DATASETS = {
    "acm": load_acm,
}


def load_dataset(name):
    name = name.lower()

    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available datasets: {list(DATASETS.keys())}"
        )

    return DATASETS[name]()

def list_datasets():
    """
    List available datasets.
    """

    return list(DATASETS.keys())