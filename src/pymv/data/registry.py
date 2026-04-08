from .acm import load_acm
from .amazon_photos import load_amazon_photos
from .bdgp import load_bdgp
from .coco_all import load_coco_all
from .coco_cross import load_coco_cross
from .dblp import load_dblp
from .esp_game import load_esp_game
from .flickr30k import load_flickr30k
from .iaprtc12 import load_iaprtc12
from .imdb import load_imdb
from .nus_wide import load_nus_wide
from .pascal import load_pascal
from .prokaryotic import load_prokaryotic
from .wiki import load_wiki

DATASETS = {
    "acm": load_acm,
    "amazon_photos": load_amazon_photos,
    "bdgp": load_bdgp,
    "coco_all": load_coco_all,
    "coco_cross": load_coco_cross,
    "dblp": load_dblp,
    "esp_game": load_esp_game,
    "flickr30k": load_flickr30k,
    "iaprtc12": load_iaprtc12,
    "imdb": load_imdb,
    "nus_wide": load_nus_wide,
    "pascal": load_pascal,
    "prokaryotic": load_prokaryotic,
    "wiki": load_wiki,
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
