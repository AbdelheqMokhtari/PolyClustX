import logging

from ._mat_utils import load_standard_mat

logger = logging.getLogger(__name__)


def load_protein_fold():

    return load_standard_mat("protein_fold", "ProteinFold.mat", n_views=12)
