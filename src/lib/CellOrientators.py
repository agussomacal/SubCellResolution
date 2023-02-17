import numpy as np

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.StencilCreators import get_fixed_stencil_values
from src.Indexers import ArrayIndexerNd


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
