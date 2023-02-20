import numpy as np

from src.Indexers import ArrayIndexerNd


def indifferent(average_values: np.ndarray, indexer: ArrayIndexerNd) -> np.ndarray:
    return np.zeros(np.shape(average_values))


def naive_piece_wise(average_values: np.ndarray, indexer: ArrayIndexerNd, eps=1e-10, min_val=0, max_val=1):
    si = np.zeros(np.shape(average_values))
    si[(average_values > min_val + eps) & (average_values < max_val - eps)] = 1
    return si
