import logging

from ._mat_utils import load_multimodal_mat

logger = logging.getLogger(__name__)


def load_movies_multimodal():

    return load_multimodal_mat("movies_multimodal", "Movies_multimodal.mat")
