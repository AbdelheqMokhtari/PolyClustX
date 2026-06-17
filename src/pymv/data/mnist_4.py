import logging

from ._mat_utils import load_standard_mat

logger = logging.getLogger(__name__)


def load_mnist_4():

    return load_standard_mat("mnist_4", "MNIST-4.mat", n_views=3)
