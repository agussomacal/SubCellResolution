from math import comb
from typing import Generator, Dict, Tuple

import numpy as np

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.AuxiliaryStructures.PolynomialAuxiliaryFunctions import fit_polynomial_from_integrals, \
    evaluate_polynomial_integral_in_rectangle, evaluate_polynomial
from lib.CellCreators.CellCreatorBase import CellCreatorBase, CellBase, REGULAR_CELL_TYPE
# ======================================== #
#           Regular cells
# ======================================== #
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import ArrayIndexerNd
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


def weight_cells_by_smoothness(central_cell_coords: int, average_values: np.ndarray, cells_smoothness: np.ndarray,
                               num_coefs: int,
                               central_cell_importance: float = 0, epsilon: float = 1e-5, delta: float = 0):
    """

    :param central_cell_coords:
    :param average_values:
    :param cells_smoothness:
    :param num_coefs:
    :param central_cell_importance:
        - 0 means it weight equally to others, no extra is added.
        - 1, 2 means it weights the double, triple if all weight equal.
    :param epsilon: to avoid division by 0 in the index I if smoothness = 0.
    :param delta:
        - 0 means only takes into account the cells that are near (in average value) to the central cell
        - otherwise it weights them like an step function 1-delta and delta. 0.5 delta means no distinction.
    :return:
    """
    if epsilon < np.inf:
        I = 1.0 / (cells_smoothness + epsilon)
        I /= np.sqrt(np.sum(I ** 2))  # normalized to 1
        I *= len(average_values)  # normalized so each cell weight one in case of equal smoothness
    else:
        I = 1.0
    N = np.sign(np.argsort((average_values - average_values[central_cell_coords]) ** 2) - num_coefs - 0.5)
    # N = (1 - N)/2 + delta * N
    N = 1 / 2 + (delta - 1 / 2) * N
    weight = I * N
    weight[central_cell_coords] += central_cell_importance
    return weight


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
        degree = min((self.degree, polynomial_max_degree))
        if self.weight_function is None:
            weight = None
        else:
            weight = self.weight_function(
                central_cell_coords=np.where(~np.any(indexer[stencil.coords - coords.array[np.newaxis, :]], axis=0))[0][
                    0],
                average_values=np.array([average_values[indexer[c]] for c in stencil.coords]),
                cells_smoothness=np.array([smoothness_index[indexer[c]] for c in stencil.coords]),
                num_coefs=(1 + degree) ** self.dimensionality if self.full_rank else comb(
                    degree + self.dimensionality, degree))
        polynomial_coefs = fit_polynomial_from_integrals(
            rectangles=[np.array([c, c + 1]) for c in stencil.coords],
            values=stencil.values,
            degree=degree,
            sample_weight=weight,
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
