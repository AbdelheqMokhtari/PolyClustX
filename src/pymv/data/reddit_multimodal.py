import logging

from ._mat_utils import load_multimodal_mat

logger = logging.getLogger(__name__)


def load_reddit_multimodal():

    return load_multimodal_mat("reddit_multimodal", "Reddit_multimodal.mat")
