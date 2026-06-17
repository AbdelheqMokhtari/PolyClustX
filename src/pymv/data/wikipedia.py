import logging

from ._mat_utils import load_standard_mat

logger = logging.getLogger(__name__)


def load_wikipedia():

    return load_standard_mat("wikipedia", "Wikipedia-test.mat", n_views=2)
