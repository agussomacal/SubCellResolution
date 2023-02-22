import operator
from typing import Tuple

import numpy as np

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords


def get_coordinates(array) -> Tuple:
    return tuple(map(np.ravel, np.meshgrid(*list(map(range, np.shape(array))))))


def iterate_all(smoothness_index, **kwargs):
    x, y = get_coordinates(smoothness_index)
    for coords in zip(x, y):
        yield CellCoords(coords)


def iterate_by_condition_on_smoothness(smoothness_index, value=0, condition: operator = operator.ge, **kwargs):
    for coords in zip(*get_coordinates(smoothness_index)):
        if condition(smoothness_index[coords], value):
            yield CellCoords(coords)


def iterate_by_smoothness(smoothness_index: np.ndarray, **kwargs):
    """
    Iterate from higher values of smoothness to lower ones.
    :param smoothness_index:
    :param kwargs:
    :return:
    """
    coords = get_coordinates(smoothness_index)
    for i in np.argsort(-smoothness_index.ravel()):
        yield CellCoords(tuple([int(c[i]) for c in coords]))


def iterate_by_reconstruction_error(reconstruction_error: np.ndarray, **kwargs):
    coords = get_coordinates(reconstruction_error)
    for i in np.argsort(-reconstruction_error.ravel()):
        yield CellCoords(tuple([int(c[i]) for c in coords]))
