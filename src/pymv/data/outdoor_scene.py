import logging

from ._mat_utils import load_standard_mat

logger = logging.getLogger(__name__)


def load_outdoor_scene():

    return load_standard_mat("outdoor_scene", "OutdoorScene.mat", n_views=4)
