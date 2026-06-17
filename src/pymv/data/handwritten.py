import logging

from ._mat_utils import load_standard_mat

logger = logging.getLogger(__name__)


def load_handwritten():

    return load_standard_mat("handwritten", "Handwritten.mat", n_views=6)
