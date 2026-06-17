from .acm import load_acm
from .acm_5view import load_acm_5view
from .aloi import load_aloi
from .amazon_photos import load_amazon_photos
from .arts_multimodal import load_arts_multimodal
from .bbc4view import load_bbc4view
from .bbcsport import load_bbcsport
from .bdgp import load_bdgp
from .caltech101_7 import load_caltech101_7
from .citeseer import load_citeseer
from .coco_all import (
    load_coco_all,
    load_coco_all_clip,
    load_coco_all_resnet_minilm,
    load_coco_all_vit_minilm,
    load_coco_all_vit_sbert,
)
from .coco_cross import (
    load_coco_cross,
    load_coco_cross_clip,
    load_coco_cross_resnet_minilm,
    load_coco_cross_vit_minilm,
    load_coco_cross_vit_sbert,
)
from .cora import load_cora
from .dataset import MultiViewDataset
from .dblp import load_dblp
from .esp_game import load_esp_game
from .flickr30k import load_flickr30k
from .grocery_multimodal import load_grocery_multimodal
from .grocerys_multimodal import load_grocerys_multimodal
from .handwritten import load_handwritten
from .hundred_leaves import load_100leaves
from .iaprtc12 import load_iaprtc12
from .imdb import load_imdb
from .mnist_10k import load_mnist_10k
from .mnist_4 import load_mnist_4
from .movies_multimodal import load_movies_multimodal
from .msrc_v5 import load_msrc_v5
from .nus_wide import load_nus_wide
from .outdoor_scene import load_outdoor_scene
from .pascal import (
    load_pascal,
    load_pascal_clip,
    load_pascal_resnet_minilm,
    load_pascal_vit_minilm,
    load_pascal_vit_sbert,
)
from .photo_multimodal import load_photo_multimodal
from .prokaryotic import load_prokaryotic
from .protein_fold import load_protein_fold
from .reddit_multimodal import load_reddit_multimodal
from .reddits_multimodal import load_reddits_multimodal
from .registry import list_datasets, load_dataset
from .reuters_1200 import load_reuters_1200
from .reuters_1500 import load_reuters_1500
from .three_sources import load_3sources
from .toys_multimodal import load_toys_multimodal
from .uci import load_uci
from .webkb import load_webkb
from .wiki import load_wiki
from .wikipedia import load_wikipedia
from .yale import load_yale

__all__ = [
    "MultiViewDataset",
    "load_dataset",
    "list_datasets",
    # Original datasets
    "load_acm",
    "load_amazon_photos",
    "load_bdgp",
    "load_coco_all",
    "load_coco_all_clip",
    "load_coco_all_resnet_minilm",
    "load_coco_all_vit_minilm",
    "load_coco_all_vit_sbert",
    "load_coco_cross",
    "load_coco_cross_clip",
    "load_coco_cross_resnet_minilm",
    "load_coco_cross_vit_minilm",
    "load_coco_cross_vit_sbert",
    "load_dblp",
    "load_esp_game",
    "load_flickr30k",
    "load_iaprtc12",
    "load_imdb",
    "load_nus_wide",
    "load_pascal",
    "load_pascal_clip",
    "load_pascal_resnet_minilm",
    "load_pascal_vit_minilm",
    "load_pascal_vit_sbert",
    "load_prokaryotic",
    "load_wiki",
    # Standard multi-view
    "load_100leaves",
    "load_3sources",
    "load_acm_5view",
    "load_aloi",
    "load_bbc4view",
    "load_bbcsport",
    "load_caltech101_7",
    "load_citeseer",
    "load_cora",
    "load_handwritten",
    "load_mnist_10k",
    "load_mnist_4",
    "load_msrc_v5",
    "load_outdoor_scene",
    "load_protein_fold",
    "load_reuters_1200",
    "load_reuters_1500",
    "load_uci",
    "load_webkb",
    "load_wikipedia",
    "load_yale",
    # Multimodal
    "load_arts_multimodal",
    "load_grocery_multimodal",
    "load_grocerys_multimodal",
    "load_movies_multimodal",
    "load_photo_multimodal",
    "load_reddit_multimodal",
    "load_reddits_multimodal",
    "load_toys_multimodal",
]
