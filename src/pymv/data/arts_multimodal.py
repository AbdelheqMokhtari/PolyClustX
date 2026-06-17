import logging

from ._mat_utils import load_multimodal_mat

logger = logging.getLogger(__name__)


def load_arts_multimodal():

    return load_multimodal_mat("arts_multimodal", "Arts_clip_multimodal.mat")
