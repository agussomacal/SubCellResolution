from typing import Tuple

import numpy as np

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords


def get_coordinates(array) -> Tuple:
    return tuple(map(np.ravel, np.meshgrid(*list(map(range, np.shape(array))))))


def iterate_default(smoothness_index: np.ndarray, reconstruction_error: np.ndarray):
    x, y = get_coordinates(smoothness_index)
    for coords in zip(x, y):
        yield CellCoords(coords)


def iterate_by_smoothness(smoothness_index: np.ndarray, reconstruction_error: np.ndarray):
    x, y = get_coordinates(smoothness_index)
    for i in np.argsort(-smoothness_index.ravel()):
        yield CellCoords((x[i], y[i]))


def iterate_by_reconstruction_error(smoothness_index: np.ndarray, reconstruction_error: np.ndarray):
    x, y = get_coordinates(reconstruction_error)
    for i in np.argsort(-reconstruction_error.ravel()):
        yield CellCoords((x[i], y[i]))
