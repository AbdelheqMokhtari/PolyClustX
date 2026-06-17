import logging

from ._mat_utils import load_standard_mat

logger = logging.getLogger(__name__)


def load_msrc_v5():

    return load_standard_mat("msrc_v5", "MSRC-v5.mat", n_views=5)
