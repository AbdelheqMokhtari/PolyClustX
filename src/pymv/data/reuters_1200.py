import logging

from ._mat_utils import load_standard_mat

logger = logging.getLogger(__name__)


def load_reuters_1200():

    return load_standard_mat("reuters_1200", "Reuters-1200.mat", n_views=5)
