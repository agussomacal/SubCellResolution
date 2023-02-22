import numpy as np

from lib.CellIterators import get_coordinates
from lib.CellOrientators import approximate_gradient_by
from lib.StencilCreators import get_fixed_stencil_values
from src.Indexers import ArrayIndexerNd


def indifferent(average_values: np.ndarray, indexer: ArrayIndexerNd) -> np.ndarray:
    return np.zeros(np.shape(average_values))


def naive_piece_wise(average_values: np.ndarray, indexer: ArrayIndexerNd, eps=1e-10, min_val=0, max_val=1):
    si = np.zeros(np.shape(average_values))
    si[(average_values > min_val + eps) & (average_values < max_val - eps)] = 1
    return si


def by_gradient(average_values: np.ndarray, indexer: ArrayIndexerNd):
    si = np.zeros(np.shape(average_values))
    x, y = get_coordinates(average_values)
    for coords in zip(x, y):
        gradient = approximate_gradient_by(
            average_values=get_fixed_stencil_values(stencil_size=(3, 3), coords=coords, average_values=average_values,
                                                    indexer=indexer),
            method="scharr",
            normalize=False
        )
        si[coords] = np.sqrt(np.sum(gradient**2))
    return si
