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
from .reuters_1200 import load_reuters_1200
from .reuters_1500 import load_reuters_1500
from .three_sources import load_3sources
from .toys_multimodal import load_toys_multimodal
from .uci import load_uci
from .webkb import load_webkb
from .wiki import load_wiki
from .wikipedia import load_wikipedia
from .yale import load_yale

DATASETS = {
    # ── Original datasets ─────────────────────────────────────────
    "acm": load_acm,
    "amazon_photos": load_amazon_photos,
    "bdgp": load_bdgp,
    "coco_all": load_coco_all,
    "coco_all_clip": load_coco_all_clip,
    "coco_all_resnet_minilm": load_coco_all_resnet_minilm,
    "coco_all_vit_minilm": load_coco_all_vit_minilm,
    "coco_all_vit_sbert": load_coco_all_vit_sbert,
    "coco_cross": load_coco_cross,
    "coco_cross_clip": load_coco_cross_clip,
    "coco_cross_resnet_minilm": load_coco_cross_resnet_minilm,
    "coco_cross_vit_minilm": load_coco_cross_vit_minilm,
    "coco_cross_vit_sbert": load_coco_cross_vit_sbert,
    "dblp": load_dblp,
    "esp_game": load_esp_game,
    "flickr30k": load_flickr30k,
    "iaprtc12": load_iaprtc12,
    "imdb": load_imdb,
    "nus_wide": load_nus_wide,
    "pascal": load_pascal,
    "pascal_clip": load_pascal_clip,
    "pascal_resnet_minilm": load_pascal_resnet_minilm,
    "pascal_vit_minilm": load_pascal_vit_minilm,
    "pascal_vit_sbert": load_pascal_vit_sbert,
    "prokaryotic": load_prokaryotic,
    "wiki": load_wiki,
    # ── Standard multi-view ───────────────────────────────────────
    "100leaves": load_100leaves,
    "3sources": load_3sources,
    "acm_5view": load_acm_5view,
    "aloi": load_aloi,
    "bbc4view": load_bbc4view,
    "bbcsport": load_bbcsport,
    "caltech101_7": load_caltech101_7,
    "citeseer": load_citeseer,
    "cora": load_cora,
    "handwritten": load_handwritten,
    "mnist_10k": load_mnist_10k,
    "mnist_4": load_mnist_4,
    "msrc_v5": load_msrc_v5,
    "outdoor_scene": load_outdoor_scene,
    "protein_fold": load_protein_fold,
    "reuters_1200": load_reuters_1200,
    "reuters_1500": load_reuters_1500,
    "uci": load_uci,
    "webkb": load_webkb,
    "wikipedia": load_wikipedia,
    "yale": load_yale,
    # ── Multimodal ────────────────────────────────────────────────
    "arts_multimodal": load_arts_multimodal,
    "grocery_multimodal": load_grocery_multimodal,
    "grocerys_multimodal": load_grocerys_multimodal,
    "movies_multimodal": load_movies_multimodal,
    "photo_multimodal": load_photo_multimodal,
    "reddit_multimodal": load_reddit_multimodal,
    "reddits_multimodal": load_reddits_multimodal,
    "toys_multimodal": load_toys_multimodal,
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
