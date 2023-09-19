from typing import Dict, Tuple, Callable

import numpy as np

from lib.AuxiliaryStructures.Constants import NEIGHBOURHOOD_8_MANHATTAN
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.CellCreators.CellCreatorBase import CellBase, REGULAR_CELL_TYPE
from lib.CellOrientators import approximate_gradient_by
from lib.StencilCreators import get_fixed_stencil_values, StencilCreatorFixedShape
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import ArrayIndexerNd


def get_opposite_regular_cells_by_stencil(coords: CellCoords, cells: Dict[Tuple[int], CellBase],
                                          independent_axis: int,
                                          average_values: np.ndarray, smoothness_index: np.ndarray,
                                          indexer: ArrayIndexerNd, stencil_size=5, **kwargs):
    stencil = StencilCreatorFixedShape(stencil_shape=(stencil_size, stencil_size)).get_stencil(
        average_values=average_values, smoothness_index=smoothness_index, coords=coords,
        independent_axis=independent_axis, indexer=indexer)
    mask = [indexer[coords_i] in cells.keys() and (cells[indexer[coords_i]].CELL_TYPE == REGULAR_CELL_TYPE) for coords_i
            in stencil.coords]

    eps = 1e-10
    dealignement = stencil.coords[mask, 1 - independent_axis] - coords[1 - independent_axis]
    regular_opposite_cell_coords = (stencil.coords[mask][np.argmin(stencil.values[mask] + eps * dealignement)],
                                    stencil.coords[mask][np.argmax(stencil.values[mask] - eps * dealignement)])

    # order from down to up given dependant axis.
    regular_opposite_cell_coords = sorted(regular_opposite_cell_coords, key=lambda c: c[1 - independent_axis])
    return tuple([cells[tuple(indexer[o])] for o in regular_opposite_cell_coords])


def get_regular_opposite_cell_coords_by_direction(coords: CellCoords, cells: Dict[Tuple[int, ...], CellBase],
                                                  average_values: np.ndarray, indexer: ArrayIndexerNd,
                                                  direction: np.ndarray,
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
            coords_i = 1 * coords.coords + 0.5 + i * direction * sign
            coords_i = np.array(coords_i, dtype=int)
            if acceptance_criterion(coords_i):
                regular_opposite_cells.append(tuple(coords_i))
                break
            else:
                singular_cells.add(tuple(coords_i))
    # sort from low to up.
    # regular_opposite_cells = sorted(regular_opposite_cells, key=lambda c: average_values[c])
    return regular_opposite_cells, singular_cells


def get_regular_opposite_cell_coords_by_minmax(coords: CellCoords, average_values: np.ndarray, indexer: ArrayIndexerNd,
                                               acceptance_criterion: Callable, direction=np.array([1, 0])) \
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

    neighbours = np.array([direction, -direction, direction[::-1], -direction[::-1]])

    regular_opposite_cells = [coords.array, coords.array]
    singular_cells = set()
    for f, mmfunc in enumerate([np.argmax, np.argmin]):
        for _ in np.arange(np.shape(average_values)[0]):
            new_coords = regular_opposite_cells[f] + neighbours
            winner = mmfunc([average_values[indexer[c]] for c in new_coords])
            singular_cells.add(tuple(regular_opposite_cells[f].tolist()))
            regular_opposite_cells[f] = new_coords[winner]
            if acceptance_criterion(regular_opposite_cells[f]):
                break
    return regular_opposite_cells, singular_cells


# def get_regular_opposite_cell_coords_sorted(coords: CellCoords, average_values: np.ndarray, dependent_axis: int,
#                                             regularity_mask: np.ndarray, indexer: ArrayIndexerNd) \
#         -> (Tuple[Tuple[int, int], Tuple[int, int]], np.ndarray):
#     regular_opposite_cells, singular_cells = get_regular_opposite_cell_coords(coords, dependent_axis, regularity_mask,
#                                                                               indexer)
#     regular_opposite_cells = np.transpose(indexer[regular_opposite_cells])
#     stencil_order = np.argsort([average_values[tuple(coord)] for coord in regular_opposite_cells])
#     return regular_opposite_cells[stencil_order], singular_cells


def get_opposite_cells_by_smoothness_threshold(coords: CellCoords, cells: Dict[Tuple[int], CellBase],
                                               independent_axis: int,
                                               average_values: np.ndarray, smoothness_index: np.ndarray,
                                               indexer: ArrayIndexerNd,
                                               threshold=0.5, **kwargs):
    regular_opposite_cell_coords, _ = get_regular_opposite_cell_coords_by_direction(
        coords=coords, cells=cells, average_values=average_values, indexer=indexer,
        direction=np.array([0, 1])[[independent_axis, 1 - independent_axis]],
        acceptance_criterion=lambda coords_i: indexer[coords_i] in cells.keys() and
                                              (cells[indexer[coords_i]].CELL_TYPE == REGULAR_CELL_TYPE) and
                                              smoothness_index[indexer[coords_i]] <= threshold,
        start=2)

    # regular_opposite_cell_coords, singular_cells = get_regular_opposite_cell_coords(
    #     coords, 1-independent_axis, smoothness_index >= threshold, indexer)
    # regular_opposite_cell_coords = np.transpose(indexer[regular_opposite_cell_coords])
    # stencil_order = np.argsort([average_values[tuple(coord)] for coord in regular_opposite_cell_coords])
    # regular_opposite_cell_coords = regular_opposite_cell_coords[stencil_order]
    # regular_opposite_cell_coords, _ = get_regular_opposite_cell_coords_sorted(coords, average_values, 1-independent_axis,
    #                                                                           smoothness_index >= threshold, indexer)

    # order from down to up given dependant axis.
    regular_opposite_cell_coords = sorted(regular_opposite_cell_coords, key=lambda c: c[1 - independent_axis])
    return tuple([cells[tuple(indexer[o])] for o in regular_opposite_cell_coords])


def get_opposite_regular_cells_by_minmax(coords: CellCoords, cells: Dict[Tuple[int], CellBase],
                                         independent_axis: int, average_values: np.ndarray,
                                         indexer: ArrayIndexerNd, direction="vertical", **kwargs):
    """

    Given the direction of the dependent axis will go up and down searching for the two regular cells.
    :param coords:
    :param cells:
    :param independent_axis:
    :param average_values:
    :param smoothness_index:
    :param indexer:
    :param kwargs:
    :return:
    """
    if direction == "grad":
        direction = approximate_gradient_by(
            average_values=get_fixed_stencil_values(stencil_size=(3, 3), coords=coords, average_values=average_values,
                                                    indexer=indexer),
            method="scharr",
            normalize=True
        ) / 2
    elif direction == "vertical":
        direction = np.array([0, 1])
    else:
        raise Exception(f"Direction {direction} not implemented.")

    regular_opposite_cell_coords, _ = get_regular_opposite_cell_coords_by_minmax(
        coords=coords, average_values=average_values, indexer=indexer,
        direction=direction[[independent_axis, 1 - independent_axis]],
        acceptance_criterion=lambda coords_i: indexer[coords_i] in cells.keys() and
                                              (cells[indexer[coords_i]].CELL_TYPE == REGULAR_CELL_TYPE))

    # order from down to up given dependant axis.
    regular_opposite_cell_coords = sorted(regular_opposite_cell_coords, key=lambda c: c[1 - independent_axis])
    return tuple([cells[tuple(indexer[o])] for o in regular_opposite_cell_coords])


def get_opposite_regular_cells(coords: CellCoords, cells: Dict[Tuple[int], CellBase],
                               independent_axis: int,
                               average_values: np.ndarray, smoothness_index: np.ndarray,
                               indexer: ArrayIndexerNd, direction="vertical", **kwargs):
    """

    Given the direction of the dependent axis will go up and down searching for the two regular cells.
    :param coords:
    :param cells:
    :param independent_axis:
    :param average_values:
    :param smoothness_index:
    :param indexer:
    :param kwargs:
    :return:
    """
    if direction == "grad":
        direction = approximate_gradient_by(
            average_values=get_fixed_stencil_values(stencil_size=(3, 3), coords=coords, average_values=average_values,
                                                    indexer=indexer),
            method="scharr",
            normalize=True
        ) / 2
    elif direction == "vertical":
        direction = np.array([0, 1])
    else:
        raise Exception(f"Direction {direction} not implemented.")

    regular_opposite_cell_coords, _ = get_regular_opposite_cell_coords_by_direction(
        coords=coords, cells=cells, average_values=average_values, indexer=indexer,
        direction=direction[[independent_axis, 1 - independent_axis]],
        acceptance_criterion=lambda coords_i: indexer[coords_i] in cells.keys() and
                                              (cells[indexer[coords_i]].CELL_TYPE == REGULAR_CELL_TYPE),
        start=1)

    # order from down to up given dependant axis.
    regular_opposite_cell_coords = sorted(regular_opposite_cell_coords, key=lambda c: c[1 - independent_axis])
    return tuple([cells[tuple(indexer[o])] for o in regular_opposite_cell_coords])


def get_opposite_cells_by_relative_smoothness(coords: CellCoords, cells: Dict[Tuple[int], CellBase],
                                              independent_axis: int,
                                              average_values: np.ndarray, smoothness_index: np.ndarray,
                                              indexer: ArrayIndexerNd,
                                              threshold=0.5, **kwargs):
    regular_opposite_cell_coords, _ = get_regular_opposite_cell_coords_by_direction(
        coords=coords, cells=cells, average_values=average_values, indexer=indexer,
        direction=np.array([0, 1])[[independent_axis, 1 - independent_axis]],
        acceptance_criterion=lambda coords_i: indexer[coords_i] in cells.keys() and
                                              (cells[indexer[coords_i]].CELL_TYPE == REGULAR_CELL_TYPE) and
                                              smoothness_index[indexer[coords_i]] <= threshold * smoothness_index[
                                                  indexer[coords.tuple]],
        start=2)

    # order from down to up given dependant axis.
    regular_opposite_cell_coords = sorted(regular_opposite_cell_coords, key=lambda c: c[1 - independent_axis])
    return tuple([cells[tuple(indexer[o])] for o in regular_opposite_cell_coords])


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
                               average_values: np.ndarray, smoothness_index, indexer: ArrayIndexerNd,
                               stencils: Dict[Tuple[int, ...], np.ndarray], **kwargs):
    gradient = approximate_gradient_by(
        average_values=get_fixed_stencil_values(stencil_size=(3, 3), coords=coords, average_values=average_values,
                                                indexer=indexer),
        method="scharr",
        normalize=True
    )
    regular_opposite_cell_coords, _ = get_regular_opposite_cell_coords_by_direction(
        coords=coords, cells=cells, average_values=average_values, indexer=indexer, direction=gradient,
        acceptance_criterion=
        lambda coords_i:
        indexer[coords_i] in cells.keys() and  # cell exists
        (cells[indexer[coords_i]].CELL_TYPE == REGULAR_CELL_TYPE) and  # cell is regular
        (coords.tuple not in set(map(tuple, stencils[indexer[coords_i]].tolist()))) and  # stencil is not polluted
        (smoothness_index[indexer[coords_i]] < 0.5 * smoothness_index[indexer[coords.tuple]]),
        start=2)

    # TODO: repeated code
    # order from down to up given dependant axis.
    regular_opposite_cell_coords = sorted(regular_opposite_cell_coords, key=lambda c: c[1 - independent_axis])
    return tuple([cells[tuple(indexer[o])] for o in regular_opposite_cell_coords])
