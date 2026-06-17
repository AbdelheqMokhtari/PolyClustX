import logging

from ._mat_utils import load_standard_mat

logger = logging.getLogger(__name__)


def load_100leaves():

    return load_standard_mat("100leaves", "100Leaves.mat", n_views=3)
