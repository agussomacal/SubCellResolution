from math import comb
from typing import Generator, Dict, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.AuxiliaryStructures.PolynomialAuxiliaryFunctions import fit_polynomial_from_integrals, \
    evaluate_polynomial_integral_in_rectangle, evaluate_polynomial, monomials_integral_in_rectangles_iterator
from lib.CellCreators.CellCreatorBase import CellCreatorBase, CellBase, REGULAR_CELL_TYPE
# ======================================== #
#           Regular cells
# ======================================== #
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import ArrayIndexerNd
from lib.CellCreators.RegularCellCreator import PolynomialCell
from src.lib.StencilCreators import Stencil


class WENO16RegularCellCreator(CellCreatorBase):
    @staticmethod
    def get_4x4_in9_2x2_decomposition(avg):
        return np.array([[- (3 * avg[0, 1]) / 4 - (3 * avg[1, 0]) / 4 + (9 * avg[1, 1]) / 4 + avg[0, 0] / 4,
                          (3 * avg[1, 2]) / 4 + (3 * avg[1, 1]) / 4 - avg[0, 2] / 4 - avg[0, 1] / 4,
                          - (3 * avg[0, 2]) / 4 - (3 * avg[1, 3]) / 4 + (9 * avg[1, 2]) / 4 + avg[0, 3] / 4],

                         [(3 * avg[2, 1]) / 4 - avg[2, 0] / 4 + (3 * avg[1, 1]) / 4 - avg[1, 0] / 4,
                          avg[2, 2] / 4 + avg[2, 1] / 4 + avg[1, 2] / 4 + avg[1, 1] / 4,
                          - avg[2, 3] / 4 + (3 * avg[2, 2]) / 4 - avg[1, 3] / 4 + (3 * avg[1, 2]) / 4],

                         [- (3 * avg[2, 0]) / 4 - (3 * avg[3, 1]) / 4 + avg[3, 0] / 4 + (9 * avg[2, 1]) / 4,
                          - avg[3, 2] / 4 - avg[3, 1] / 4 + (3 * avg[2, 2]) / 4 + (3 * avg[2, 1]) / 4,
                          - (3 * avg[2, 3]) / 4 - (3 * avg[3, 2]) / 4 + avg[3, 3] / 4 + (9 * avg[2, 2]) / 4]])

    @staticmethod
    def get_4x4_in9_2x2_smoothness_index(smoothness_index):
        Iij = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                Iij[i, j] = np.mean(smoothness_index[i:i + 2, j:j + 2])
        return Iij

    @staticmethod
    def get_polynomial_from4eval_points_and_avg(coords, average_values, eval_points):
        e = eval_points
        # avg = average_values[coords.tuple]
        x0, y0 = coords.tuple
        c = np.zeros((2, 2))
        c[0, 0] = e[0, 0] * x0 * y0 + e[0, 0] * x0 + e[0, 0] * y0 + e[0, 0] - e[0, 1] * x0 * y0 - e[0, 1] * y0 - e[
            1, 0] * x0 * y0 - e[1, 0] * x0 + e[1, 1] * x0 * y0
        c[0, 1] = -e[0, 0] * x0 - e[0, 0] + e[0, 1] * x0 + e[0, 1] + e[1, 0] * x0 - e[1, 1] * x0
        c[1, 0] = -e[0, 0] * y0 - e[0, 0] + e[0, 1] * y0 + e[1, 0] * y0 + e[1, 0] - e[1, 1] * y0
        c[1, 1] = e[0, 0] - e[0, 1] - e[1, 0] + e[1, 1]

        # c[0, 0] = -6 * avg * x0 ** 2 - 6 * avg * x0 + e[0, 0] * x0 ** 2 / 2 + e[0, 0] * x0 * y0 + 3 * e[0, 0] * x0 / 2 + \
        #           e[0, 0] * y0 ** 2 + 2 * e[0, 0] * y0 + e[0, 0] + 5 * e[0, 1] * x0 ** 2 / 2 - e[0, 1] * x0 * y0 + 5 * \
        #           e[0, 1] * x0 / 2 - e[0, 1] * y0 ** 2 - 2 * e[0, 1] * y0 + 5 * e[1, 0] * x0 ** 2 / 2 \
        #           - e[1, 0] * x0 * y0 + 3 * e[1, 0] * x0 / 2 - e[1, 0] * y0 ** 2 - e[1, 0] * y0 + e[1, 1] * x0 ** 2 / 2 \
        #           + e[1, 1] * x0 * y0 + e[1, 1] * x0 / 2 + e[1, 1] * y0 ** 2 + e[1, 1] * y0
        # c[0, 1] = -e[0, 0] * x0 - 2 * e[0, 0] * y0 - 2 * e[0, 0] + e[0, 1] * x0 + 2 * e[0, 1] * y0 + 2 * e[0, 1] + e[
        #     1, 0] * x0 + 2 * e[1, 0] * y0 + e[1, 0] - e[1, 1] * x0 - 2 * e[1, 1] * y0 - e[1, 1]
        # c[0, 2] = e[0, 0] - e[0, 1] - e[1, 0] + e[1, 1]
        # c[1, 0] = 12 * avg * x0 + 6 * avg - e[0, 0] * x0 - e[0, 0] * y0 - 3 * e[0, 0] / 2 - 5 * e[0, 1] * x0 + e[
        #     0, 1] * y0 - 5 * e[0, 1] / 2 - 5 * e[1, 0] * x0 + e[1, 0] * y0 - 3 * e[1, 0] / 2 - e[1, 1] * x0 - e[
        #               1, 1] * y0 - \
        #           e[1, 1] / 2
        # c[1, 1] = e[0, 0] - e[0, 1] - e[1, 0] + e[1, 1]
        # c[2, 0] = -6 * avg + e[0, 0] / 2 + 5 * e[0, 1] / 2 + 5 * e[1, 0] / 2 + e[1, 1] / 2
        return c

    def create_cells(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                     coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                     stencil: Stencil, stencils: Dict[Tuple[int, ...], np.ndarray]) -> Generator[CellBase, None, None]:
        assert np.shape(stencil.values) == (25,), \
            f"Shape of stencil should be 5x5 but found {np.shape(stencil.values)}"
        avg = np.zeros((5, 5))
        I = np.zeros((5, 5))
        min_coords = np.min(stencil.coords, axis=0)
        for c in stencil.coords:
            ix = tuple(np.array(c) - min_coords)
            I[ix] = smoothness_index[indexer[c]]
            avg[ix] = average_values[indexer[c]]

        gamma = np.array([[1, 4, 1],
                          [4, 16, 4],
                          [1, 4, 1]]) / 36

        point_eval_approx = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                poly_evals_3x3 = self.get_4x4_in9_2x2_decomposition(avg[i:i + 4, j:j + 4])
                Iij = self.get_4x4_in9_2x2_smoothness_index(I[i:i + 4, j:j + 4])
                alpha = gamma / (1 + Iij) ** 2
                w = alpha / np.sum(alpha)
                point_eval_approx[i, j] = np.sum(poly_evals_3x3 * w)

        polynomial_coefs = self.get_polynomial_from4eval_points_and_avg(coords, average_values, point_eval_approx)

        yield PolynomialCell(coords, polynomial_coefs)
