import logging

from ._mat_utils import load_standard_mat

logger = logging.getLogger(__name__)


def load_caltech101_7():

    return load_standard_mat("caltech101_7", "Caltech101-7.mat", n_views=6)
