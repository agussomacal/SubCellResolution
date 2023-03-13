from typing import Generator, Dict, Tuple

import numpy as np

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.AuxiliaryStructures.PolynomialAuxiliaryFunctions import fit_polynomial_from_integrals, \
    evaluate_polynomial_integral_in_rectangle, evaluate_polynomial
from lib.CellCreators.CellCreatorBase import CellCreatorBase, CellBase, REGULAR_CELL_TYPE
# ======================================== #
#           Regular cells
# ======================================== #
from src.Indexers import ArrayIndexerNd
from src.lib.StencilCreators import Stencil


class CellRegularBase(CellBase):
    CELL_TYPE = REGULAR_CELL_TYPE


class PolynomialCell(CellRegularBase):
    def __init__(self, coords: CellCoords, polynomial_coefs):
        super().__init__(coords)
        self.polynomial_coefs = polynomial_coefs

    def integrate_rectangle(self, rectangle) -> float:
        return evaluate_polynomial_integral_in_rectangle(self.polynomial_coefs, [rectangle])[0]

    def evaluate(self, query_points: np.ndarray) -> np.ndarray:
        return evaluate_polynomial(self.polynomial_coefs, query_points)

    def __str__(self):
        return super(PolynomialCell, self).__str__() + "degree {}".format(np.shape(self.polynomial_coefs))


# ======================================== #
#           Regular cell creators
# ======================================== #

def weight_cells_extra_weight(is_central_cell, smoothness_index_i, central_cell_extra_weight=0):
    return (1 + central_cell_extra_weight * is_central_cell)


def weight_cells(is_central_cell, smoothness_index_i, central_cell_extra_weight=0):
    return (1 + central_cell_extra_weight * is_central_cell) / (1 + smoothness_index_i)


class PolynomialRegularCellCreator(CellCreatorBase):
    def __init__(self, degree, dimensionality=2, noisy=False, weight_function=None, full_rank=False):
        """
        If there is presence of noise in the cell averages then a leastsq fit will be done instead of an interpolation
        and it will be done with an odd number of cells in each dimension to have symetry that why (2*noise+1). If noise
        is Flase == 0 then regular interpolation otherwise extending the grid a smoother fit.
        """
        self.degree = degree
        self.dimensionality = dimensionality
        self.noisy = noisy
        self.weight_function = weight_function
        self.full_rank = full_rank
        # only dimension 1, needs to know the problem dimensionality
        super().__init__()

    def create_cells(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                     coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                     stencil: Stencil, stencils: Dict[Tuple[int, ...], np.ndarray]) -> Generator[CellBase, None, None]:
        # (self.polynomial_max_degree + 1) * (1 + noisy)
        polynomial_max_degree = int(np.floor(len(stencil.coords) ** (1 / self.dimensionality)) / (1 + self.noisy) - 1)
        polynomial_coefs = fit_polynomial_from_integrals(
            rectangles=[np.array([c, c + 1]) for c in stencil.coords],
            values=stencil.values,
            degree=min((self.degree, polynomial_max_degree)),
            sample_weight=None if self.weight_function is None else [
                self.weight_function(coords.tuple == indexer[c], smoothness_index[indexer[c]]) for c in stencil.coords],
            full_rank=self.full_rank
        )
        yield PolynomialCell(coords, polynomial_coefs)


class PiecewiseConstantRegularCellCreator(CellCreatorBase):
    def __init__(self, apriori_up_value, apriori_down_value, dimensionality=2):
        """
        If there is presence of noise in the cell averages then a leastsq fit will be done instead of an interpolation
        and it will be done with an odd number of cells in each dimension to have symetry that why (2*noise+1). If noise
        is Flase == 0 then regular interpolation otherwise extending the grid a smoother fit.
        """
        assert apriori_up_value > apriori_down_value, "apriori_up_value should be greater than apriori_down_value"
        self.apriori_values = [apriori_down_value, apriori_up_value]
        self.midpoint = (apriori_up_value - apriori_down_value) / 2
        self.dimensionality = dimensionality
        # only dimension 1, needs to know the problem dimensionality
        super().__init__()

    def create_cells(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                     coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                     stencil: Stencil, stencils: Dict[Tuple[int, ...], np.ndarray]) -> Generator[CellBase, None, None]:
        yield PolynomialCell(
            coords,
            np.reshape(self.apriori_values[np.mean(stencil.values) > self.midpoint],
                       np.repeat(1, self.dimensionality))
        )

    def __str__(self):
        return super().__str__() + "PiecewiseConstant"
