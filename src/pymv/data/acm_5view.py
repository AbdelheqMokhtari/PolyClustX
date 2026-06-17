import logging

from ._mat_utils import load_standard_mat

logger = logging.getLogger(__name__)


def load_acm_5view():

    return load_standard_mat("acm_5view", "ACM.mat", n_views=5)
