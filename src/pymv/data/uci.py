import logging

from ._mat_utils import load_standard_mat

logger = logging.getLogger(__name__)


def load_uci():

    return load_standard_mat("uci", "UCI.mat", n_views=3)
