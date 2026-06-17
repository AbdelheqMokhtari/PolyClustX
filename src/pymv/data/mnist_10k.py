import logging

from ._mat_utils import load_standard_mat

logger = logging.getLogger(__name__)


def load_mnist_10k():

    return load_standard_mat("mnist_10k", "MNIST-10k.mat", n_views=3)
