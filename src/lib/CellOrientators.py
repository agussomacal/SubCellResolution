from typing import List

import numpy as np

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.StencilCreators import get_fixed_stencil_values
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import ArrayIndexerNd

# https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/

ScharrKernel = np.array(
    [[3, 10, 3],
     [0, 0, 0],
     [-3, -10, -3]])
OptimSobel3x3 = np.array([
    [1, 3.5887, 1],
    [0, 0, 0],
    [-1, -3.5887, -1]
])
OptimSobel5x5 = np.array([
    [0.0007, 0.0052, 0.0370, 0.0052, 0.0007],
    [0.0037, 0.1187, 0.2589, 0.1187, 0.0037],
    [0, 0, 0, 0, 0],
    [-0.0037, -0.1187, -0.2589, -0.1187, -0.0037],
    [-0.0007, -0.0052, -0.0370, -0.0052, -0.0007],
])


def approximate_gradient_by(average_values, method="optim", normalize=False):
    """
    https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/
    :return:
    """
    assert np.shape(average_values) in [(3, 3), (5, 5)], "Gradient approximation only working for 3x3 stencils."
    if np.shape(average_values) == (3, 3):
        if method == "scharr":
            gx = ScharrKernel
        elif method == "optim":
            gx = OptimSobel3x3
        else:
            raise Exception("Not implemented method {}.".format(method))
    elif np.shape(average_values) == (5, 5):
        gx = OptimSobel5x5
    else:
        raise Exception("Gradient approximation only working for 3x3 or 5x5 stencils.")

    gy = gx.T
    g = np.array([np.sum(average_values * gx), np.sum(average_values * gy)])
    if normalize and np.all(g != 0):
        g /= np.sqrt(np.dot(g, g))
    return g


# ---------------------------------------------- #
# ----------------- Base class ----------------- #
class BaseOrientator:
    def __init__(self, dimensionality: int = 2):
        self.dimensionality = dimensionality

    def get_independent_axis(self, coords: CellCoords, average_values: np.ndarray, indexer: ArrayIndexerNd) -> List[
        int]:
        return [0]


class OrientPredefined(BaseOrientator):
    def __init__(self, predefined_axis, dimensionality: int = 2):
        super().__init__(dimensionality)
        self.predefined_axis = predefined_axis

    def get_independent_axis(self, coords: CellCoords, average_values: np.ndarray, indexer: ArrayIndexerNd) -> List[
        int]:
        assert len(np.shape(average_values)) == 2, "Only for 2 dimensions."
        return [self.predefined_axis]


class OrientByGradient(BaseOrientator):
    def __init__(self, kernel_size=(3, 3), dimensionality: int = 2, method="optim", angle_threshold=45):
        super().__init__(dimensionality)
        self.kernel_size = kernel_size
        self.method = method
        self.angle_threshold = angle_threshold*np.pi/180

    def get_independent_axis(self, coords: CellCoords, average_values: np.ndarray, indexer: ArrayIndexerNd) -> List[
        int]:
        assert len(np.shape(average_values)) == 2, "Only for 2 dimensions."
        stencil_values = get_fixed_stencil_values(self.kernel_size, coords, average_values, indexer)
        g = approximate_gradient_by(stencil_values, method=self.method, normalize=False)
        gx = np.abs(g[0])
        gy = np.abs(g[1])
        orientation = int(gx >= gy)
        # if approximated angle is lower than threshold propose only one orientation
        return [orientation] if np.arctan(min(gy, gx) / (max(gx, gy) + 1e-15)) <= self.angle_threshold \
            else [orientation, 1 - orientation]
