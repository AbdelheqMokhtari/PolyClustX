import logging

from ._mat_utils import load_standard_mat

logger = logging.getLogger(__name__)


def load_reuters_1500():

    return load_standard_mat("reuters_1500", "Reuters-1500.mat", n_views=5)
