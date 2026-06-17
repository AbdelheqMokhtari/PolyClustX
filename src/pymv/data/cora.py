import logging

from ._mat_utils import load_standard_mat

logger = logging.getLogger(__name__)


def load_cora():

    return load_standard_mat("cora", "Cora.mat", n_views=4)
