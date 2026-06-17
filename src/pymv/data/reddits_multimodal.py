import logging

from ._mat_utils import load_multimodal_mat

logger = logging.getLogger(__name__)


def load_reddits_multimodal():

    return load_multimodal_mat("reddits_multimodal", "RedditS_multimodal.mat")
