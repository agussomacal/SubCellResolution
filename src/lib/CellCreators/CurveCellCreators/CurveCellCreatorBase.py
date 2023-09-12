from collections import namedtuple
from typing import Tuple, Dict, Generator, Callable, Union

import numpy as np

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.CellCreators.CellCreatorBase import CellCreatorBase, \
    CellBase, CURVE_CELL_TYPE
from lib.Curves.Curves import Curve
from lib.StencilCreators import Stencil
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import ArrayIndexerNd


def get_x_points(stencil, independent_axis):
    return 0.5 + np.arange(np.min(stencil.coords, axis=0)[independent_axis],
                           np.max(stencil.coords, axis=0)[independent_axis] + 1)


def map2unidimensional(value_up, value_down, independent_axis: int, stencil: Stencil) -> [np.ndarray, np.ndarray]:
    dependent_axis = 1 - independent_axis
    # if the values are not 0 or 1
    min_value = np.min((value_up, value_down))
    shape = np.max(stencil.coords, axis=0) - np.min(stencil.coords, axis=0) + 1
    stencil_values = stencil.values.reshape(shape).sum(axis=dependent_axis) - shape[dependent_axis] * min_value

    # which side the integral has to be done, is 0 below the curve or is 1?
    jump = value_up - value_down
    if jump > 0:
        stencil_values = shape[dependent_axis] * jump - stencil_values
    stencil_values += np.min(stencil.coords, axis=0)[dependent_axis]
    x_points = get_x_points(stencil, independent_axis)
    return x_points, stencil_values


def get_values_up_down(coords, regular_opposite_cells):
    value_up = regular_opposite_cells[1].evaluate(coords.coords + 0.5)
    value_down = regular_opposite_cells[0].evaluate(coords.coords + 0.5)
    return value_up, value_down


# def get_coords_up_down(coords, regular_opposite_cells: Tuple[CellBase, CellBase], stencil, smoothness_index):
#     """Regular opposite cell may not be in stencil. """
#     cells_on_each_side = []
#     for cells2visit in [[regular_opposite_cells[0].coords], [regular_opposite_cells[1].coords]]:
#         cells_on_each_side.append([])
#         while len(cells2visit) > 0:
#             next_cell = cells2visit.pop()
#             cells_on_each_side.append(next_cell.tuple)
#
#
#     value_up = regular_opposite_cells[1].evaluate(coords.coords + 0.5)
#     value_down = regular_opposite_cells[0].evaluate(coords.coords + 0.5)
#     return value_up, value_down


def prepare_stencil4one_dimensionalization(independent_axis: int, value_up: Union[int, float],
                                           value_down: Union[int, float], stencil: Stencil,
                                           smoothness_index: np.ndarray, indexer: ArrayIndexerNd):
    # if the values are not 0 or 1
    stencil_values = stencil.values - np.min((value_up, value_down))
    stencil_values /= np.max((value_up, value_down))

    # # thresholding in case of piecewise-regular
    # stencil_values[stencil_values < 0] = 0
    # stencil_values[stencil_values > 1] = 1

    # reshape stencil in rectangular form
    ks = np.max(stencil.coords, axis=0) - np.min(stencil.coords, axis=0) + 1
    stencil_values = stencil.values.reshape(ks)
    stencil_values = np.transpose(stencil_values, [independent_axis, 1 - independent_axis])

    # thresholding in case of piecewise-regular
    # assumes that smoothness comes with 1 for
    stencil_smoothness = np.reshape([smoothness_index[indexer[coord]] for coord in stencil.coords], ks)
    stencil_smoothness = np.transpose(stencil_smoothness, [independent_axis, 1 - independent_axis])
    assert (np.sum(stencil_smoothness == 0) + np.sum(stencil_smoothness == 1)) == np.prod(
        ks), "Only works if smoothness has 1 for curve cells and 0 otherwise."
    below_ix = stencil_smoothness.cumsum(axis=1) == 0
    above_ix = (stencil_smoothness.cumsum(axis=1) * (1 - stencil_smoothness)) > 0
    stencil_values[below_ix] = 0
    stencil_values[above_ix] = 1

    # one-dimensional versioin with 1 "down" and 0 "up"
    stencil_values = (1 - stencil_values) if value_up > value_down else stencil_values
    return stencil_values


# ====================================================== #
#                    Curve cell Base                     #
# ====================================================== #
def get_zone(curve, point, independent_axis, dependent_axis):
    y = curve.function(point[independent_axis])
    if isinstance(y, np.ndarray):
        # when the curve is not a function (like a circle) crosses many times, then it will belong to one zone or
        # the other depending on the position with respect to the multiple function values.
        # That's why sorting the values then seeing where to insert and modulus 2 for counting changes.
        # TODO: This fails if we are in a transition a value of y with 0 vertical derivative of the curve.
        return np.searchsorted(np.sort(np.squeeze(y)), point[dependent_axis]) % 2
    return 1 * (y < point[dependent_axis])


class CellCurveBase(CellBase):
    CELL_TYPE = CURVE_CELL_TYPE

    def __init__(self, coords: CellCoords, curve: Curve, regular_opposite_cells: Tuple[CellBase, CellBase],
                 dependent_axis: int = 1):
        super().__init__(coords)

        assert len(self.coords.coords) < 3, "This method only works in 2D"
        self._curve = curve
        self.dependent_axis = dependent_axis
        self.independent_axis = 1 - dependent_axis

        # evaluates where the regular cells are positioned with respect of the curve, if above or below.
        self.regular_opposite_cells = regular_opposite_cells
        self.zone_roc0 = get_zone(self.curve, self.regular_opposite_cells[0].coords.tuple, self.independent_axis,
                                  self.dependent_axis)

    @property
    def curve(self):
        return self._curve

    @curve.setter
    def curve(self, curve):
        self._curve = curve
        self.zone_roc0 = get_zone(self._curve, self.regular_opposite_cells[0].coords.tuple, self.independent_axis,
                                  self.dependent_axis)

    def __str__(self):
        return super(CellCurveBase, self).__str__() + str(self.curve)

    def integrate_rectangle(self, rectangle) -> float:
        return self.curve.calculate_rectangle_average(
            x_limits=rectangle[:, self.independent_axis],
            y_limits=rectangle[:, self.dependent_axis]
        )

    def evaluate(self, query_points: np.ndarray) -> np.ndarray:
        return self.curve.evaluate(x=query_points[:, self.independent_axis],
                                   y=query_points[:, self.dependent_axis])
        # return np.array([self.regular_opposite_cells[
        #                      (get_zone(self.curve, point, self.independent_axis,
        #                                self.dependent_axis) - self.zone_roc0) % 2].evaluate(point)
        #                  for point in query_points])


# ====================================================== #
#                  Curve cell creator base               #
# ====================================================== #
CurveAlternative = namedtuple("CurveAlternative", ["cell", "stencil"])


class CurveCellCreatorBase(CellCreatorBase):
    def __init__(self, regular_opposite_cell_searcher: Callable):
        self.regular_opposite_cell_searcher = regular_opposite_cell_searcher

    def create_cells(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                     coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                     stencil: Stencil, stencils: Dict[Tuple[int, ...], np.ndarray]) -> Generator[CellBase, None, None]:
        regular_opposite_cells = self.regular_opposite_cell_searcher(
            coords=coords, independent_axis=independent_axis, average_values=average_values,
            smoothness_index=smoothness_index, indexer=indexer, cells=cells, stencil=stencil,
            stencils=stencils
        )
        if len(regular_opposite_cells) == 2:
            for curve in self.create_curves(average_values, indexer, cells, coords, smoothness_index,
                                            independent_axis, stencil, regular_opposite_cells):
                yield CellCurveBase(
                    coords=coords,
                    curve=curve,
                    regular_opposite_cells=regular_opposite_cells,
                    dependent_axis=1 - independent_axis)

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[Curve, None, None]:
        raise Exception("Not implemented.")
