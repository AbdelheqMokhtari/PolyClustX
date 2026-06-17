import logging

from ._mat_utils import load_standard_mat

logger = logging.getLogger(__name__)


def load_3sources():

    return load_standard_mat("3sources", "3Sources.mat", n_views=3)
