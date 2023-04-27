import numpy as np

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.StencilCreators import get_fixed_stencil_values
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import ArrayIndexerNd


# https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/


def approximate_gradient_by(average_values, method="scharr", normalize=False):
    """
    https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/
    :return:
    """
    if method == "scharr":
        scharr_gy = np.array(
            [[3, 0, -3],
             [10, 0, -10],
             [3, 0, -3]])
        scharr_gx = np.array(
            [[3, 10, 3],
             [0, 0, 0],
             [-3, -10, -3]])
        g = np.array([np.sum(average_values * scharr_gx), np.sum(average_values * scharr_gy)])
    elif method == "sobel":
        raise Exception("Not implemented method {}.".format(method))
    else:
        raise Exception("Not implemented method {}.".format(method))
    if normalize and np.all(g != 0):
        g /= np.sqrt(np.dot(g, g))
    return g


# ---------------------------------------------- #
# ----------------- Base class ----------------- #
class BaseOrientator:
    def __init__(self, dimensionality: int = 2):
        self.dimensionality = dimensionality

    def get_independent_axis(self, coords: CellCoords, average_values: np.ndarray, indexer: ArrayIndexerNd) -> int:
        return 0


class OrientPredefined(BaseOrientator):
    def __init__(self, predefined_axis, dimensionality: int = 2):
        super().__init__(dimensionality)
        self.predefined_axis = predefined_axis

    def get_independent_axis(self, coords: CellCoords, average_values: np.ndarray, indexer: ArrayIndexerNd) -> int:
        assert len(np.shape(average_values)) == 2, "Only for 2 dimensions."
        return self.predefined_axis


class OrientByGradient(BaseOrientator):
    def __init__(self, kernel_size=(3, 3), dimensionality: int = 2):
        super().__init__(dimensionality)
        self.kernel_size = kernel_size

    def get_independent_axis(self, coords: CellCoords, average_values: np.ndarray, indexer: ArrayIndexerNd) -> int:
        assert len(np.shape(average_values)) == 2, "Only for 2 dimensions."
        stencil = get_fixed_stencil_values(self.kernel_size, coords, average_values, indexer)
        independent_axis = int(np.abs(np.diff(stencil, axis=0)).sum() >= np.abs(np.diff(stencil, axis=1)).sum())
        return independent_axis
