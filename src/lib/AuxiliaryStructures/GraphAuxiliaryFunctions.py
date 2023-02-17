import itertools
from typing import Tuple, Set, List

import numpy as np

from src.Indexers import ArrayIndexerNd


def cell_neighbours(central_ix: Tuple, indexer: ArrayIndexerNd = None, stencil_radius=1) -> List[Tuple[int]]:
    """

    :param central_ix: (i, ); (i, j); (i, j, k) ...
    :param stencil_radius: 1 -> is a singular_neighbours of 3^dim centred in the central_ix
    :return: list of indexes in singular_neighbours sorted from near to farther.
    """
    stencil_radius = [stencil_radius] * len(central_ix) if isinstance(stencil_radius, int) else stencil_radius

    return sorted({coords if indexer is None else indexer[coords] for coords in itertools.product(
        *[np.arange(-sr, sr + 1, dtype=int) + ix for ix, sr in zip(central_ix, stencil_radius)])},
                  key=lambda x: np.sum((np.array(x) - np.array(central_ix)) ** 2))[1:]


def mesh_iterator(mesh_size, out_type=tuple):
    for ix in itertools.product(*list(map(np.arange, mesh_size))):
        yield out_type(ix)


def get_1d_around_cells(ix, axis, length=1, sides="both"):
    assert sides in ["both", "positive", "negative"], "sides must be in both, positive, negative"
    num_sides = 2 if sides == "both" else 1
    new_cells = np.array([ix] * num_sides * length).T
    new_cells[axis] += np.array(np.append(
        np.arange(-length, 0) if sides in ["both", "negative"] else [],
        np.arange(1, length + 1) if sides in ["both", "positive"] else []
    ), dtype=int)
    return list(map(tuple, new_cells.T))


def add_coord2index(index, value, axis):
    ix_next = list(index)
    ix_next[axis] += value
    return tuple(ix_next)
