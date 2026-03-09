from .registry import load_dataset, list_datasets
from .acm import load_acm
from .amazon_photos import load_amazon_photos
from .bdgp import load_bdgp
from .coco_all import load_coco_all
from .coco_cross import load_coco_cross
from .dblp import load_dblp
from .imdb import load_imdb
from .pascal import load_pascal
from .prokaryotic import load_prokaryotic
from .wiki import load_wiki

__all__ = [
    "load_dataset",
    "list_datasets",
    "load_acm",
    "load_amazon_photos",
    "load_bdgp",
    "load_coco_all",
    "load_coco_cross",
    "load_dblp",
    "load_imdb",
    "load_pascal",
    "load_prokaryotic",
    "load_wiki",
]