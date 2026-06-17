import logging

from ._mat_utils import load_multimodal_mat

logger = logging.getLogger(__name__)


def load_toys_multimodal():

    return load_multimodal_mat("toys_multimodal", "Toys_multimodal.mat")
