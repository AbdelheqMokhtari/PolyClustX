import logging

from ._mat_utils import load_standard_mat

logger = logging.getLogger(__name__)


def load_bbcsport():

    return load_standard_mat("bbcsport", "BBCSport.mat", n_views=2)
