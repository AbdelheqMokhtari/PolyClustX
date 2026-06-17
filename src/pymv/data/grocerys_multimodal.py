import logging

from ._mat_utils import load_multimodal_mat

logger = logging.getLogger(__name__)


def load_grocerys_multimodal():

    return load_multimodal_mat("grocerys_multimodal", "GroceryS_multimodal.mat")
