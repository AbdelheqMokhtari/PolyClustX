import logging

from ._mat_utils import load_multimodal_mat

logger = logging.getLogger(__name__)


def load_grocery_multimodal():

    return load_multimodal_mat("grocery_multimodal", "Grocery_multimodal.mat")
