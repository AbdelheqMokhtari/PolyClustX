import logging

from ._mat_utils import load_standard_mat

logger = logging.getLogger(__name__)


def load_yale():

    return load_standard_mat("yale", "Yale.mat", n_views=3)
