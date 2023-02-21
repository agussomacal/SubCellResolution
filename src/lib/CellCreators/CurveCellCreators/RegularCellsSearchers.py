from typing import Dict, Tuple, Callable

import numpy as np

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.CellCreators.CellCreatorBase import CellBase, REGULAR_CELL_TYPE
from lib.CellOrientators import approximate_gradient_by
from lib.StencilCreators import get_fixed_stencil_values
from src.Indexers import ArrayIndexerNd


def get_regular_opposite_cell_coords(coords: CellCoords, cells: Dict[Tuple[int, ...], CellBase],
                                     average_values: np.ndarray, indexer: ArrayIndexerNd, direction: np.ndarray,
                                     acceptance_criterion: Callable, start=2) \
        -> (Tuple[Tuple[int, int], Tuple[int, int]], set):
    """

    :param coords:
    :param cells:
    :param average_values:
    :param indexer:
    :param direction:
    :param start: to avoid searching to near the singular cell at the beginning.
    :return:
    """
    regular_opposite_cells = []
    singular_cells = {coords.tuple}
    for sign in [1, -1]:
        for i in np.arange(start, np.shape(average_values)[0]):
            coords_i = 1 * coords.coords + i * direction * sign
            coords_i = np.array(coords_i, dtype=int)
            # if cells[indexer[coords_i]].CELL_TYPE == REGULAR_CELL:
            if acceptance_criterion(coords_i):
                regular_opposite_cells.append(tuple(coords_i))
                break
            else:
                singular_cells.add(tuple(coords_i))
    # sort from low to up.
    # regular_opposite_cells = sorted(regular_opposite_cells, key=lambda c: average_values[c])
    return regular_opposite_cells, singular_cells


# def get_regular_opposite_cell_coords_sorted(coords: CellCoords, average_values: np.ndarray, dependent_axis: int,
#                                             regularity_mask: np.ndarray, indexer: ArrayIndexerNd) \
#         -> (Tuple[Tuple[int, int], Tuple[int, int]], np.ndarray):
#     regular_opposite_cells, singular_cells = get_regular_opposite_cell_coords(coords, dependent_axis, regularity_mask,
#                                                                               indexer, direction=[])
#     regular_opposite_cells = np.transpose(indexer[regular_opposite_cells])
#     stencil_order = np.argsort([average_values[tuple(coord)] for coord in regular_opposite_cells])
#     return regular_opposite_cells[stencil_order], singular_cells


def get_opposite_cells_by_smoothness_threshold(coords: CellCoords, cells: Dict[Tuple[int], CellBase],
                                               independent_axis: int,
                                               average_values: np.ndarray, smoothness_index: np.ndarray,
                                               indexer: ArrayIndexerNd,
                                               threshold=0.5, **kwargs):
    regular_opposite_cell_coords, _ = get_regular_opposite_cell_coords(
        coords=coords, cells=cells, average_values=average_values, indexer=indexer,
        direction=np.array([1, 0])[[independent_axis, 1 - independent_axis]],
        acceptance_criterion=lambda coords_i: indexer[coords_i] in cells.keys() and
                                              smoothness_index[indexer[coords_i]] <= threshold,
        start=2)
    # regular_opposite_cell_coords, _ = get_regular_opposite_cell_coords_sorted(coords, average_values, dependent_axis,
    #                                                                           smoothness_index >= threshold, indexer)

    regular_opposite_cells = tuple([cells[tuple(o)] for o in regular_opposite_cell_coords])
    # order from down to up given dependant axis.
    return tuple(sorted(regular_opposite_cells, key=lambda roc: roc.coords[1 - independent_axis]))


# def get_smooth_opposite_cells(coords: CellCoords, dependent_axis: int, average_values: np.ndarray,
#                               smoothness_index, indexer: ArrayIndexerNd,
#                               cells: Dict[Tuple[int], CellBase], stencil, **kwargs):
#     # stencil_coords = cell_neighbours(central_ix=coords.tuple, stencil_radius=2)  # 5x5x..x5 stencil
#
#     sc = sorted(stencil_coords, key=lambda c: smoothness_index[indexer[c]])
#     first_neighbour = sc.pop(0)
#     central_cell_value = average_values[indexer[coords.tuple]]
#     first_neighbour_sign = np.sign(average_values[indexer[first_neighbour]] - central_cell_value)
#     sc = list(filter(lambda c: np.sign(average_values[indexer[c]] - central_cell_value) != first_neighbour_sign, sc))
#     second_neighbour = sc.pop(0)
#     regular_opposite_cell_coords = [indexer[first_neighbour], indexer[second_neighbour]]
#
#     # TODO: repeated code
#     regular_opposite_cells = tuple([cells[tuple(o)] for o in regular_opposite_cell_coords])
#     # order from down to up given dependant axis.
#     return tuple(sorted(regular_opposite_cells, key=lambda roc: roc.coords[dependent_axis]))


def get_opposite_cells_by_grad(coords: CellCoords, cells: Dict[Tuple[int], CellBase], independent_axis: int,
                               average_values: np.ndarray, smoothness_index, indexer: ArrayIndexerNd, **kwargs):
    # stencil_coords = cell_neighbours(central_ix=coords.tuple, stencil_radius=2)  # 5x5x..x5 stencil
    gradient = approximate_gradient_by(
        average_values=get_fixed_stencil_values(stencil_size=(3, 3), coords=coords, average_values=average_values,
                                                indexer=indexer),
        method="scharr",
        normalize=True
    )
    regular_opposite_cell_coords, _ = get_regular_opposite_cell_coords(
        coords=coords, cells=cells, average_values=average_values, indexer=indexer, direction=gradient,
        acceptance_criterion=
        lambda coords_i: indexer[coords_i] in cells.keys() and
                         (cells[indexer[coords_i]].CELL_TYPE == REGULAR_CELL_TYPE) and
                         (smoothness_index[indexer[coords_i]] < smoothness_index[indexer[coords.tuple]]),
        start=2)

    # sc = sorted(stencil_coords, key=lambda c: smoothness_index[indexer[c]])
    # first_neighbour = sc.pop(0)
    # central_cell_value = average_values[indexer[coords.tuple]]
    # first_neighbour_sign = np.sign(average_values[indexer[first_neighbour]] - central_cell_value)
    # sc = list(filter(lambda c: np.sign(average_values[indexer[c]] - central_cell_value) != first_neighbour_sign, sc))
    # second_neighbour = sc.pop(0)
    # regular_opposite_cell_coords = [indexer[first_neighbour], indexer[second_neighbour]]

    # TODO: repeated code
    regular_opposite_cells = tuple([cells[tuple(o)] for o in regular_opposite_cell_coords])
    # order from down to up given dependant axis.
    # return tuple(sorted(regular_opposite_cells, key=lambda roc: roc.coords[1 - independent_axis]))
    return tuple(sorted(regular_opposite_cells, key=lambda roc: -average_values[roc.coords.tuple]))
