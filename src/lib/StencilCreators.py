import itertools
from collections import namedtuple
from typing import Tuple, Union, List, Callable

import numpy as np

from lib.AuxiliaryStructures.Constants import REGULAR_CELL, NEIGHBOURHOOD_8_MANHATTAN
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from src.Indexers import ArrayIndexerNd, CYCLIC

Stencil = namedtuple("Stencil", "coords values")


def rotate_matrix_90deg(matrix, times=1):
    for _ in range(times % 4):
        matrix = matrix.T[::-1, :]
    return matrix


def get_stencil_pre_cart_prod(stencil_size: Union[np.ndarray, Tuple[int, int]],
                              coords: Union[Tuple[int, int], CellCoords], indexer: ArrayIndexerNd) -> List[CellCoords]:
    return [indexer.transform_single_dimension(np.arange(size) - size // 2 + coords[i], i) for i, size in
            enumerate(stencil_size)]


def get_fixed_stencil_values(stencil_size: Union[np.ndarray, Tuple[int, int]],
                             coords: Union[Tuple[int, int], CellCoords],
                             average_values: np.ndarray, indexer: ArrayIndexerNd) -> np.ndarray:
    return average_values[np.ix_(*get_stencil_pre_cart_prod(stencil_size, coords, indexer))]


# ---------------------------------------------- #
# ----------------- Base class ----------------- #
class StencilCreator:
    def get_stencil(self, average_values: np.ndarray, smoothness_index: np.ndarray, coords: CellCoords,
                    independent_axis: int, indexer: ArrayIndexerNd) -> Stencil:
        """
        Gives the stencil without passing through the indexer.
        """
        raise Exception("Not implemented.")


# ----------------------------------------------- #
# ----------------- Fixed shape ----------------- #
class StencilCreatorFixedShape(StencilCreator):
    def __init__(self, stencil_shape: Tuple[int, int]):
        self.stencil_shape = np.array(stencil_shape)

    def get_stencil(self, average_values: np.ndarray, smoothness_index: np.ndarray, coords: CellCoords,
                    independent_axis: int, indexer: ArrayIndexerNd) -> Stencil:
        stencil_coords = np.array([c for c in itertools.product(*[np.arange(size) - size // 2 + coords[i] for i, size in
                                                                  enumerate(self.stencil_shape[
                                                                                [independent_axis,
                                                                                 1 - independent_axis]])])])
        stencil_values = get_fixed_stencil_values(self.stencil_shape[[independent_axis, 1 - independent_axis]], coords,
                                                  average_values, indexer)
        return Stencil(coords=stencil_coords, values=stencil_values)


# --------------------------------------------- #
# ----------------- Same type ----------------- #
def get_stencil_same_type(coords: CellCoords, indexer: ArrayIndexerNd, num_nodes: int, cell_mask: np.ndarray):
    stencil = []
    nodes2visit = [coords]
    for ix in range(num_nodes):
        visiting_coords = nodes2visit.pop(0)
        yield visiting_coords
        stencil.append(visiting_coords)
        new_node_candidates = {
            CellCoords(indexer[new_coords.coords]) for new_coords in visiting_coords + NEIGHBOURHOOD_8_MANHATTAN
            if cell_mask[indexer[new_coords.tuple]] == cell_mask[visiting_coords.tuple]
        }
        nodes2visit += list(new_node_candidates.difference(stencil + nodes2visit))
        if len(nodes2visit) == 0:
            break


def get_neighbouring_singular_coords_under_condition(coords: CellCoords, indexer: ArrayIndexerNd, cell_mask: np.ndarray,
                                                     condition: Callable, max_num_nodes=5):
    cell_type = cell_mask[coords.tuple]
    visited_nodes = [
        CellCoords(indexer[new_coords.coords]) for new_coords in coords + NEIGHBOURHOOD_8_MANHATTAN
        if cell_mask[indexer[new_coords.tuple]] == cell_type
    ]

    if len(visited_nodes) < 2:
        return []
    else:
        stop = [False, False]
        nodes2visit = [[visited_nodes.pop(-1)], [visited_nodes.pop(-1)]]
        for ix in range(max_num_nodes):
            for direction in [0, 1]:
                if not stop[direction]:
                    visiting_coords = nodes2visit[direction].pop(0)
                    if condition(visiting_coords):
                        yield visiting_coords
                        stop[direction] = True
                    else:
                        visited_nodes.append(visiting_coords)
                        new_node_candidates = {
                            CellCoords(indexer[new_coords.coords]) for new_coords in
                            visiting_coords + NEIGHBOURHOOD_8_MANHATTAN
                            if cell_mask[indexer[new_coords.tuple]] == cell_type
                        }
                        nodes2visit[direction] += list(
                            new_node_candidates.difference(visited_nodes + nodes2visit[0] + nodes2visit[1]))
                        if len(nodes2visit) == 0:
                            stop[direction] = True


class StencilCreatorSameRegionAdaptive(StencilCreator):
    def __init__(self, num_nodes_per_dim: int, dimensionality: int):
        self.num_nodes = num_nodes_per_dim ** dimensionality

    def get_stencil(self, average_values: np.ndarray, smoothness_index: np.ndarray, coords: CellCoords,
                    independent_axis: int, indexer: ArrayIndexerNd) -> Stencil:
        scoords = list(get_stencil_same_type(coords, indexer, self.num_nodes, cell_mask=smoothness_index))
        values = np.array([average_values[c.tuple] for c in scoords])
        return Stencil(coords=scoords, values=values)


# -------------------------------------------------- #
# ----------------- Adaptive shape ----------------- #
def get_regular_opposite_cell_coords(coords: CellCoords, dependent_axis: int, regularity_mask: np.ndarray,
                                     indexer: ArrayIndexerNd) -> (Tuple[Tuple[int, int], Tuple[int, int]], np.ndarray):
    assert all([mode == CYCLIC for mode in indexer.modes]), "Only for periodic condition problems."
    regular_opposite_cells = []
    singular_cells = [coords.tuple]
    for direction in [1, -1]:
        for i in np.arange(np.shape(regularity_mask)[dependent_axis]):
            coords_i = 1 * coords.coords
            coords_i[dependent_axis] += direction * i
            if regularity_mask[indexer[coords_i]] == REGULAR_CELL:
                regular_opposite_cells.append(tuple(coords_i))
                break
            else:
                singular_cells.append(tuple(coords_i))

    return regular_opposite_cells, singular_cells


def get_regular_opposite_cell_coords_sorted(coords: CellCoords, average_values: np.ndarray, dependent_axis: int,
                                            regularity_mask: np.ndarray, indexer: ArrayIndexerNd) \
        -> (Tuple[Tuple[int, int], Tuple[int, int]], np.ndarray):
    regular_opposite_cells, singular_cells = get_regular_opposite_cell_coords(coords, dependent_axis, regularity_mask,
                                                                              indexer)
    regular_opposite_cells = np.transpose(indexer[regular_opposite_cells])
    stencil_order = np.argsort([average_values[tuple(coord)] for coord in regular_opposite_cells])
    return regular_opposite_cells[stencil_order], singular_cells
    # return tuple([regular_opposite_cells[o] for o in stencil_order]), singular_cells


def surf_step(coords: CellCoords, surfers_coords: np.ndarray, surfers_nearness: float, regularity_mask: np.ndarray,
              # region_mask: np.ndarray,
              regular_cells_set: np.ndarray, regular_cells_nearness: List[float], dependent_axis: int,
              surfer_direction: int, indexer: ArrayIndexerNd, weight_of_growing_on_dependent_direction=0.45):
    is_surfer_regular = regularity_mask[indexer[surfers_coords]] == REGULAR_CELL
    if np.all(is_surfer_regular):  # and not np.equal(*region_mask[indexer[surfers_coords]]):
        regular_cells_set = np.concatenate([regular_cells_set, surfers_coords])  # add the coords to the set
        regular_cells_nearness += [surfers_nearness, surfers_nearness]
        surfers_coords[:, 1 - dependent_axis] += surfer_direction  # moves 1 step in the surfer's direction 1 or -1
        surfers_nearness += 1
    else:
        # goes one step further from the center cell looking for a regular cell
        surfers_coords[~is_surfer_regular, dependent_axis] += np.sign(
            surfers_coords[~is_surfer_regular, dependent_axis] - coords[dependent_axis])
        surfers_nearness += weight_of_growing_on_dependent_direction
    return surfers_coords, surfers_nearness, regular_cells_set, regular_cells_nearness


SURFERS_DIRECTIONS = np.array([-1, 1])


class StencilCreatorAdaptive(StencilCreator):
    def __init__(self, smoothness_threshold: float, independent_dim_stencil_size: int,
                 weight_of_growing_on_dependent_direction=0.45):
        self.smoothness_threshold = smoothness_threshold
        self.independent_dim_stencil_size = independent_dim_stencil_size
        self.weight_of_growing_on_dependent_direction = weight_of_growing_on_dependent_direction

    def get_stencil_boundaries(self, regularity_mask: np.ndarray,
                               # region_mask: np.ndarray,
                               coords: CellCoords, indexer: ArrayIndexerNd, independent_axis: int = 0):
        dependent_axis = 1 - independent_axis
        regular_opposite_cells, _ = get_regular_opposite_cell_coords(coords, dependent_axis, regularity_mask, indexer)

        regular_cells_set = np.array(regular_opposite_cells)
        regular_cells_nearness = [0, 0]
        surfers = np.array([regular_opposite_cells, regular_opposite_cells])
        surfers[0, :, independent_axis] += SURFERS_DIRECTIONS[0]
        surfers[1, :, independent_axis] += SURFERS_DIRECTIONS[1]
        surfers_nearness = [1, 1]
        for _ in range((self.independent_dim_stencil_size - 1) * 2):
            for i, surfer_direction in enumerate(SURFERS_DIRECTIONS):
                surfers[i], surfers_nearness[i], regular_cells_set, regular_cells_nearness = \
                    surf_step(coords, surfers[i], surfers_nearness[i], regularity_mask,
                              # region_mask,
                              regular_cells_set,
                              regular_cells_nearness, dependent_axis, surfer_direction, indexer,
                              self.weight_of_growing_on_dependent_direction)
        regular_cells_set = regular_cells_set[np.argsort(regular_cells_nearness)][
                            :self.independent_dim_stencil_size * 2]
        stencil_boundaries = np.transpose([np.min(regular_cells_set, axis=0), np.max(regular_cells_set, axis=0)])
        return stencil_boundaries

    def get_stencil(self, average_values: np.ndarray, smoothness_index: np.ndarray, coords: CellCoords,
                    independent_axis: int, indexer: ArrayIndexerNd) -> Stencil:
        stencil_boundaries = self.get_stencil_boundaries(smoothness_index > self.smoothness_threshold, coords, indexer,
                                                         independent_axis)
        # No cyclic transformation of indexes is applied because the coordinates need to be contiguous
        # to have a correct fit
        coords = np.array(
            [c for c in itertools.product(*[np.arange(mn, mx + 1) for i, (mn, mx) in enumerate(stencil_boundaries)])])
        values = average_values[np.ix_(*[indexer.transform_single_dimension(np.arange(mn, mx + 1), i)
                                         for i, (mn, mx) in enumerate(stencil_boundaries)])]
        return Stencil(coords=coords, values=values)
