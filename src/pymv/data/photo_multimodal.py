import logging

from ._mat_utils import load_multimodal_mat

logger = logging.getLogger(__name__)


def load_photo_multimodal():

    return load_multimodal_mat("photo_multimodal", "Photo_multimodal.mat")
