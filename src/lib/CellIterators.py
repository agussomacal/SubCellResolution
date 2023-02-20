import operator
from typing import Tuple

import numpy as np

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords


def get_coordinates(array) -> Tuple:
    return tuple(map(np.ravel, np.meshgrid(*list(map(range, np.shape(array))))))


def iterate_all(smoothness_index, reconstruction_error, *args, **kwargs):
    x, y = get_coordinates(smoothness_index)
    for coords in zip(x, y):
        yield CellCoords(coords)


def iterate_by_condition_on_smoothness(smoothness_index, reconstruction_error, value=0,
                                       condition: operator = operator.ge, *args, **kwargs):
    x, y = get_coordinates(smoothness_index)
    for coords in zip(x, y):
        if condition(smoothness_index[coords], value):
            yield CellCoords(coords)


def iterate_default(smoothness_index: np.ndarray, reconstruction_error: np.ndarray, smoothness_threshold=0,
                    reconstruction_error_threshold=0):
    x, y = get_coordinates(smoothness_index)
    for coords in zip(x, y):
        if smoothness_index[coords] >= smoothness_threshold and \
                reconstruction_error[coords] >= reconstruction_error_threshold:
            yield CellCoords(coords)


def iterate_by_smoothness(smoothness_index: np.ndarray, reconstruction_error: np.ndarray, smoothness_threshold=0,
                          reconstruction_error_threshold=0, smoothness_operator=operator.ge, re_operator=operator.ge):
    x, y = get_coordinates(smoothness_index)
    for i in np.argsort(-smoothness_index.ravel()):
        coords = x[i], y[i]
        if operator(smoothness_index[coords], smoothness_threshold) and \
                operator(reconstruction_error[coords], reconstruction_error_threshold):
            yield CellCoords(coords)


def iterate_by_reconstruction_error(smoothness_index: np.ndarray, reconstruction_error: np.ndarray,
                                    smoothness_threshold=0,
                                    reconstruction_error_threshold=0):
    x, y = get_coordinates(reconstruction_error)
    for i in np.argsort(-reconstruction_error.ravel()):
        coords = x[i], y[i]
        if smoothness_index[coords] >= smoothness_threshold and \
                reconstruction_error[coords] >= reconstruction_error_threshold:
            yield CellCoords(coords)
