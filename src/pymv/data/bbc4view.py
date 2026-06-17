import logging

from ._mat_utils import load_standard_mat

logger = logging.getLogger(__name__)


def load_bbc4view():

    return load_standard_mat("bbc4view", "BBC4view.mat", n_views=4)
