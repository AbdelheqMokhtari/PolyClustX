import logging

from ._mat_utils import load_standard_mat

logger = logging.getLogger(__name__)


def load_citeseer():

    return load_standard_mat("citeseer", "CiteSeer.mat", n_views=2)
