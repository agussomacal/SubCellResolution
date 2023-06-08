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
    def __init__(self, degree=1):
        self.degree = degree

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
    def get_4x4_in9_2x2_smoothness_index(smoothness_index, summary=np.mean):
        Iij = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                Iij[i, j] = summary(smoothness_index[i:i + 2, j:j + 2])
        return Iij

    def get_polynomial_from4eval_points_and_avg(self, coords, average_values, eval_points):
        e = eval_points
        x0, y0 = coords.tuple

        if self.degree == 1:
            c = np.zeros((2, 2))
            c[0, 0] = e[0, 0] * x0 * y0 + e[0, 0] * x0 + e[0, 0] * y0 + e[0, 0] - e[0, 1] * x0 * y0 - e[0, 1] * y0 - e[
                1, 0] * x0 * y0 - e[1, 0] * x0 + e[1, 1] * x0 * y0
            c[0, 1] = -e[0, 0] * x0 - e[0, 0] + e[0, 1] * x0 + e[0, 1] + e[1, 0] * x0 - e[1, 1] * x0
            c[1, 0] = -e[0, 0] * y0 - e[0, 0] + e[0, 1] * y0 + e[1, 0] * y0 + e[1, 0] - e[1, 1] * y0
            c[1, 1] = e[0, 0] - e[0, 1] - e[1, 0] + e[1, 1]
        elif self.degree == 2:
            avg = average_values[coords.tuple]
            c = np.zeros((3, 3))
            c[0, 0] = -6 * avg * x0 ** 2 - 6 * avg * x0 + e[0, 0] * x0 ** 2 / 2 + e[0, 0] * x0 * y0 + 3 * e[
                0, 0] * x0 / 2 + \
                      e[0, 0] * y0 ** 2 + 2 * e[0, 0] * y0 + e[0, 0] + 5 * e[0, 1] * x0 ** 2 / 2 - e[
                          0, 1] * x0 * y0 + 5 * \
                      e[0, 1] * x0 / 2 - e[0, 1] * y0 ** 2 - 2 * e[0, 1] * y0 + 5 * e[1, 0] * x0 ** 2 / 2 \
                      - e[1, 0] * x0 * y0 + 3 * e[1, 0] * x0 / 2 - e[1, 0] * y0 ** 2 - e[1, 0] * y0 + e[
                          1, 1] * x0 ** 2 / 2 \
                      + e[1, 1] * x0 * y0 + e[1, 1] * x0 / 2 + e[1, 1] * y0 ** 2 + e[1, 1] * y0
            c[0, 1] = -e[0, 0] * x0 - 2 * e[0, 0] * y0 - 2 * e[0, 0] + e[0, 1] * x0 + 2 * e[0, 1] * y0 + 2 * e[0, 1] + \
                      e[
                          1, 0] * x0 + 2 * e[1, 0] * y0 + e[1, 0] - e[1, 1] * x0 - 2 * e[1, 1] * y0 - e[1, 1]
            c[0, 2] = e[0, 0] - e[0, 1] - e[1, 0] + e[1, 1]
            c[1, 0] = 12 * avg * x0 + 6 * avg - e[0, 0] * x0 - e[0, 0] * y0 - 3 * e[0, 0] / 2 - 5 * e[0, 1] * x0 + e[
                0, 1] * y0 - 5 * e[0, 1] / 2 - 5 * e[1, 0] * x0 + e[1, 0] * y0 - 3 * e[1, 0] / 2 - e[1, 1] * x0 - e[
                          1, 1] * y0 - \
                      e[1, 1] / 2
            c[1, 1] = e[0, 0] - e[0, 1] - e[1, 0] + e[1, 1]
            c[2, 0] = -6 * avg + e[0, 0] / 2 + 5 * e[0, 1] / 2 + 5 * e[1, 0] / 2 + e[1, 1] / 2
        else:
            raise Exception(f"Not implemented degree {self.degree}")
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


class WENO1DRegularCellCreator(CellCreatorBase):
    def __init__(self, num_coeffs=4):
        self.num_coeffs = num_coeffs

    @staticmethod
    def get_4_in3_2_decomposition(avg):
        return np.array([-avg[0] / 2 + 3 * avg[1] / 2, avg[1] / 2 + avg[2] / 2, 3 * avg[2] / 2 - avg[3] / 2])

    @staticmethod
    def get_5_in3_3_decomposition(avg):
        return np.array([23 * avg[0] / 24 - 35 * avg[1] / 12 + 71 * avg[2] / 24,
                         -avg[1] / 24 + avg[2] / 12 + 23 * avg[3] / 24,
                         -avg[2] / 24 + 13 * avg[3] / 12 - avg[4] / 24])

    @staticmethod
    def get_4_in3_2_smoothness_index(smoothness_index, summary=np.max):
        return np.array([summary(smoothness_index[i:i + 1]) for i in range(3)])

    def get_polynomial_from4eval_points_and_avg(self, coords, average_values, eval_points, vertical_segments,
                                                horizontal_segments):
        e = eval_points
        x0, y0 = coords.tuple
        avg = average_values[coords.tuple]
        if self.num_coeffs == 4:
            c = np.zeros((2, 2))
            c[0, 0] = e[0, 0] * x0 * y0 + e[0, 0] * x0 + e[0, 0] * y0 + e[0, 0] - e[0, 1] * x0 * y0 - e[0, 1] * y0 - e[
                1, 0] * x0 * y0 - e[1, 0] * x0 + e[1, 1] * x0 * y0
            c[0, 1] = -e[0, 0] * x0 - e[0, 0] + e[0, 1] * x0 + e[0, 1] + e[1, 0] * x0 - e[1, 1] * x0
            c[1, 0] = -e[0, 0] * y0 - e[0, 0] + e[0, 1] * y0 + e[1, 0] * y0 + e[1, 0] - e[1, 1] * y0
            c[1, 1] = e[0, 0] - e[0, 1] - e[1, 0] + e[1, 1]
        elif self.num_coeffs == 5:
            c = np.zeros((3, 3))
            c[0, 0] = -3 * avg * x0 ** 2 - 3 * avg * x0 - 3 * avg * y0 ** 2 - 3 * avg * y0 + 3 * e[0, 0] * x0 ** 2 / 4 + \
                      e[0, 0] * x0 * y0 + 7 * e[0, 0] * x0 / 4 + 3 * e[0, 0] * y0 ** 2 / 4 + 7 * e[0, 0] * y0 / 4 + e[
                          0, 0] + 3 * e[0, 1] * x0 ** 2 / 4 - e[0, 1] * x0 * y0 + 3 * e[0, 1] * x0 / 4 + 3 * e[
                          0, 1] * y0 ** 2 / 4 - e[0, 1] * y0 / 4 + 3 * e[1, 0] * x0 ** 2 / 4 - e[1, 0] * x0 * y0 - e[
                          1, 0] * x0 / 4 + 3 * e[1, 0] * y0 ** 2 / 4 + 3 * e[1, 0] * y0 / 4 + 3 * e[
                          1, 1] * x0 ** 2 / 4 + e[1, 1] * x0 * y0 + 3 * e[1, 1] * x0 / 4 + 3 * e[
                          1, 1] * y0 ** 2 / 4 + 3 * e[1, 1] * y0 / 4
            c[0, 1] = 6 * avg * x0 + 3 * avg - 3 * e[0, 0] * x0 / 2 - e[0, 0] * y0 - 7 * e[0, 0] / 4 - 3 * e[
                0, 1] * x0 / 2 + e[0, 1] * y0 - 3 * e[0, 1] / 4 - 3 * e[1, 0] * x0 / 2 + e[1, 0] * y0 + e[
                          1, 0] / 4 - 3 * e[1, 1] * x0 / 2 - e[1, 1] * y0 - 3 * e[1, 1] / 4
            c[0, 2] = -3 * avg + 3 * e[0, 0] / 4 + 3 * e[0, 1] / 4 + 3 * e[1, 0] / 4 + 3 * e[1, 1] / 4
            c[1, 0] = 6 * avg * y0 + 3 * avg - e[0, 0] * x0 - 3 * e[0, 0] * y0 / 2 - 7 * e[0, 0] / 4 + e[
                0, 1] * x0 - 3 * e[0, 1] * y0 / 2 + e[0, 1] / 4 + e[1, 0] * x0 - 3 * e[1, 0] * y0 / 2 - 3 * e[
                          1, 0] / 4 - e[1, 1] * x0 - 3 * e[1, 1] * y0 / 2 - 3 * e[1, 1] / 4
            c[1, 1] = e[0, 0] - e[0, 1] - e[1, 0] + e[1, 1]
            c[1, 2] = 0
            c[2, 0] = -3 * avg + 3 * e[0, 0] / 4 + 3 * e[0, 1] / 4 + 3 * e[1, 0] / 4 + 3 * e[1, 1] / 4
            c[2, 1] = 0
            c[2, 2] = 0
        elif self.num_coeffs == 9:
            sv = vertical_segments
            sh = horizontal_segments
            c = np.zeros((3, 3))
            c[
                0, 0] = 36 * avg * x0 ** 2 * y0 ** 2 + 36 * avg * x0 ** 2 * y0 + 36 * avg * x0 * y0 ** 2 + 36 * avg * x0 * y0 + 9 * \
                        e[0, 0] * x0 ** 2 * y0 ** 2 + 12 * e[0, 0] * x0 ** 2 * y0 + 3 * e[0, 0] * x0 ** 2 + 12 * e[
                            0, 0] * x0 * y0 ** 2 + 16 * e[0, 0] * x0 * y0 + 4 * e[0, 0] * x0 + 3 * e[
                            0, 0] * y0 ** 2 + 4 * e[0, 0] * y0 + e[0, 0] + 9 * e[0, 1] * x0 ** 2 * y0 ** 2 + 6 * e[
                            0, 1] * x0 ** 2 * y0 + 12 * e[0, 1] * x0 * y0 ** 2 + 8 * e[0, 1] * x0 * y0 + 3 * e[
                            0, 1] * y0 ** 2 + 2 * e[0, 1] * y0 + 9 * e[1, 0] * x0 ** 2 * y0 ** 2 + 12 * e[
                            1, 0] * x0 ** 2 * y0 + 3 * e[1, 0] * x0 ** 2 + 6 * e[1, 0] * x0 * y0 ** 2 + 8 * e[
                            1, 0] * x0 * y0 + 2 * e[1, 0] * x0 + 9 * e[1, 1] * x0 ** 2 * y0 ** 2 + 6 * e[
                            1, 1] * x0 ** 2 * y0 + 6 * e[1, 1] * x0 * y0 ** 2 + 4 * e[1, 1] * x0 * y0 - 18 * sh[
                            0] * x0 ** 2 * y0 ** 2 - 18 * sh[0] * x0 ** 2 * y0 - 24 * sh[0] * x0 * y0 ** 2 - 24 * sh[
                            0] * x0 * y0 - 6 * sh[0] * y0 ** 2 - 6 * sh[0] * y0 - 18 * sh[1] * x0 ** 2 * y0 ** 2 - 18 * \
                        sh[1] * x0 ** 2 * y0 - 12 * sh[1] * x0 * y0 ** 2 - 12 * sh[1] * x0 * y0 - 18 * sv[
                            0] * x0 ** 2 * y0 ** 2 - 24 * sv[0] * x0 ** 2 * y0 - 6 * sv[0] * x0 ** 2 - 18 * sv[
                            0] * x0 * y0 ** 2 - 24 * sv[0] * x0 * y0 - 6 * sv[0] * x0 - 18 * sv[
                            1] * x0 ** 2 * y0 ** 2 - 12 * sv[1] * x0 ** 2 * y0 - 18 * sv[1] * x0 * y0 ** 2 - 12 * sv[
                            1] * x0 * y0
            c[0, 1] = -72 * avg * x0 * y0 ** 2 - 72 * avg * x0 * y0 - 36 * avg * y0 ** 2 - 36 * avg * y0 - 18 * e[
                0, 0] * x0 * y0 ** 2 - 24 * e[0, 0] * x0 * y0 - 6 * e[0, 0] * x0 - 12 * e[0, 0] * y0 ** 2 - 16 * e[
                          0, 0] * y0 - 4 * e[0, 0] - 18 * e[0, 1] * x0 * y0 ** 2 - 12 * e[0, 1] * x0 * y0 - 12 * e[
                          0, 1] * y0 ** 2 - 8 * e[0, 1] * y0 - 18 * e[1, 0] * x0 * y0 ** 2 - 24 * e[
                          1, 0] * x0 * y0 - 6 * e[1, 0] * x0 - 6 * e[1, 0] * y0 ** 2 - 8 * e[1, 0] * y0 - 2 * e[
                          1, 0] - 18 * e[1, 1] * x0 * y0 ** 2 - 12 * e[1, 1] * x0 * y0 - 6 * e[1, 1] * y0 ** 2 - 4 * e[
                          1, 1] * y0 + 36 * sh[0] * x0 * y0 ** 2 + 36 * sh[0] * x0 * y0 + 24 * sh[0] * y0 ** 2 + 24 * \
                      sh[0] * y0 + 36 * sh[1] * x0 * y0 ** 2 + 36 * sh[1] * x0 * y0 + 12 * sh[1] * y0 ** 2 + 12 * sh[
                          1] * y0 + 36 * sv[0] * x0 * y0 ** 2 + 48 * sv[0] * x0 * y0 + 12 * sv[0] * x0 + 18 * sv[
                          0] * y0 ** 2 + 24 * sv[0] * y0 + 6 * sv[0] + 36 * sv[1] * x0 * y0 ** 2 + 24 * sv[
                          1] * x0 * y0 + 18 * sv[1] * y0 ** 2 + 12 * sv[1] * y0
            c[0, 2] = 36 * avg * y0 ** 2 + 36 * avg * y0 + 9 * e[0, 0] * y0 ** 2 + 12 * e[0, 0] * y0 + 3 * e[0, 0] + 9 * \
                      e[0, 1] * y0 ** 2 + 6 * e[0, 1] * y0 + 9 * e[1, 0] * y0 ** 2 + 12 * e[1, 0] * y0 + 3 * e[
                          1, 0] + 9 * e[1, 1] * y0 ** 2 + 6 * e[1, 1] * y0 - 18 * sh[0] * y0 ** 2 - 18 * sh[
                          0] * y0 - 18 * sh[1] * y0 ** 2 - 18 * sh[1] * y0 - 18 * sv[0] * y0 ** 2 - 24 * sv[
                          0] * y0 - 6 * sv[0] - 18 * sv[1] * y0 ** 2 - 12 * sv[1] * y0
            c[1, 0] = -72 * avg * x0 ** 2 * y0 - 36 * avg * x0 ** 2 - 72 * avg * x0 * y0 - 36 * avg * x0 - 18 * e[
                0, 0] * x0 ** 2 * y0 - 12 * e[0, 0] * x0 ** 2 - 24 * e[0, 0] * x0 * y0 - 16 * e[0, 0] * x0 - 6 * e[
                          0, 0] * y0 - 4 * e[0, 0] - 18 * e[0, 1] * x0 ** 2 * y0 - 6 * e[0, 1] * x0 ** 2 - 24 * e[
                          0, 1] * x0 * y0 - 8 * e[0, 1] * x0 - 6 * e[0, 1] * y0 - 2 * e[0, 1] - 18 * e[
                          1, 0] * x0 ** 2 * y0 - 12 * e[1, 0] * x0 ** 2 - 12 * e[1, 0] * x0 * y0 - 8 * e[
                          1, 0] * x0 - 18 * e[1, 1] * x0 ** 2 * y0 - 6 * e[1, 1] * x0 ** 2 - 12 * e[
                          1, 1] * x0 * y0 - 4 * e[1, 1] * x0 + 36 * sh[0] * x0 ** 2 * y0 + 18 * sh[0] * x0 ** 2 + 48 * \
                      sh[0] * x0 * y0 + 24 * sh[0] * x0 + 12 * sh[0] * y0 + 6 * sh[0] + 36 * sh[1] * x0 ** 2 * y0 + 18 * \
                      sh[1] * x0 ** 2 + 24 * sh[1] * x0 * y0 + 12 * sh[1] * x0 + 36 * sv[0] * x0 ** 2 * y0 + 24 * sv[
                          0] * x0 ** 2 + 36 * sv[0] * x0 * y0 + 24 * sv[0] * x0 + 36 * sv[1] * x0 ** 2 * y0 + 12 * sv[
                          1] * x0 ** 2 + 36 * sv[1] * x0 * y0 + 12 * sv[1] * x0
            c[1, 1] = 144 * avg * x0 * y0 + 72 * avg * x0 + 72 * avg * y0 + 36 * avg + 36 * e[0, 0] * x0 * y0 + 24 * e[
                0, 0] * x0 + 24 * e[0, 0] * y0 + 16 * e[0, 0] + 36 * e[0, 1] * x0 * y0 + 12 * e[0, 1] * x0 + 24 * e[
                          0, 1] * y0 + 8 * e[0, 1] + 36 * e[1, 0] * x0 * y0 + 24 * e[1, 0] * x0 + 12 * e[
                          1, 0] * y0 + 8 * e[1, 0] + 36 * e[1, 1] * x0 * y0 + 12 * e[1, 1] * x0 + 12 * e[
                          1, 1] * y0 + 4 * e[1, 1] - 72 * sh[0] * x0 * y0 - 36 * sh[0] * x0 - 48 * sh[0] * y0 - 24 * sh[
                          0] - 72 * sh[1] * x0 * y0 - 36 * sh[1] * x0 - 24 * sh[1] * y0 - 12 * sh[1] - 72 * sv[
                          0] * x0 * y0 - 48 * sv[0] * x0 - 36 * sv[0] * y0 - 24 * sv[0] - 72 * sv[1] * x0 * y0 - 24 * \
                      sv[1] * x0 - 36 * sv[1] * y0 - 12 * sv[1]
            c[1, 2] = -72 * avg * y0 - 36 * avg - 18 * e[0, 0] * y0 - 12 * e[0, 0] - 18 * e[0, 1] * y0 - 6 * e[
                0, 1] - 18 * e[1, 0] * y0 - 12 * e[1, 0] - 18 * e[1, 1] * y0 - 6 * e[1, 1] + 36 * sh[0] * y0 + 18 * sh[
                          0] + 36 * sh[1] * y0 + 18 * sh[1] + 36 * sv[0] * y0 + 24 * sv[0] + 36 * sv[1] * y0 + 12 * sv[
                          1]
            c[2, 0] = 36 * avg * x0 ** 2 + 36 * avg * x0 + 9 * e[0, 0] * x0 ** 2 + 12 * e[0, 0] * x0 + 3 * e[0, 0] + 9 * \
                      e[0, 1] * x0 ** 2 + 12 * e[0, 1] * x0 + 3 * e[0, 1] + 9 * e[1, 0] * x0 ** 2 + 6 * e[
                          1, 0] * x0 + 9 * e[1, 1] * x0 ** 2 + 6 * e[1, 1] * x0 - 18 * sh[0] * x0 ** 2 - 24 * sh[
                          0] * x0 - 6 * sh[0] - 18 * sh[1] * x0 ** 2 - 12 * sh[1] * x0 - 18 * sv[0] * x0 ** 2 - 18 * sv[
                          0] * x0 - 18 * sv[1] * x0 ** 2 - 18 * sv[1] * x0
            c[2, 1] = -72 * avg * x0 - 36 * avg - 18 * e[0, 0] * x0 - 12 * e[0, 0] - 18 * e[0, 1] * x0 - 12 * e[
                0, 1] - 18 * e[1, 0] * x0 - 6 * e[1, 0] - 18 * e[1, 1] * x0 - 6 * e[1, 1] + 36 * sh[0] * x0 + 24 * sh[
                          0] + 36 * sh[1] * x0 + 12 * sh[1] + 36 * sv[0] * x0 + 18 * sv[0] + 36 * sv[1] * x0 + 18 * sv[
                          1]
            c[2, 2] = 36 * avg + 9 * e[0, 0] + 9 * e[0, 1] + 9 * e[1, 0] + 9 * e[1, 1] - 18 * sh[0] - 18 * sh[1] - 18 * \
                      sv[0] - 18 * sv[1]
        else:
            raise Exception(f"Not implemented degree {self.degree}")
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

        gamma = np.array([1 / 6, 2 / 3, 1 / 6])
        gamma2 = np.array([9 / 1840, 99 / 920, 71 / 80])

        vertical_segment_eval_approx = np.zeros((5, 2))
        for i in range(5):
            for j in range(2):
                poly_evals_1x4 = self.get_4_in3_2_decomposition(avg[i, j:j + 4])
                Ii = self.get_4_in3_2_smoothness_index(I[i, j:j + 4])
                alpha = gamma / (1 + Ii) ** 2
                w = alpha / np.sum(alpha)
                vertical_segment_eval_approx[i, j] = np.sum(poly_evals_1x4 * w)

        horizontal_segment_eval_approx = np.zeros((2, 5))
        for i in range(2):
            for j in range(5):
                poly_evals_1x4 = self.get_4_in3_2_decomposition(avg[i:i + 4, j])
                Ii = self.get_4_in3_2_smoothness_index(I[i:i + 4, j])
                alpha = gamma / (1 + Ii) ** 2
                w = alpha / np.sum(alpha)
                horizontal_segment_eval_approx[i, j] = np.sum(poly_evals_1x4 * w)

        point_eval_approx = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                poly_evals_1x4 = self.get_4_in3_2_decomposition(vertical_segment_eval_approx[i:i + 4, j])
                Ii = self.get_4_in3_2_smoothness_index(I[i:i + 4, j + 1:j + 2])
                alpha = gamma / (1 + Ii) ** 2
                w = alpha / np.sum(alpha)
                point_eval_approx[i, j] = np.sum(poly_evals_1x4 * w)

                poly_evals_1x4 = self.get_4_in3_2_decomposition(horizontal_segment_eval_approx[i, j:j + 4])
                Ii = self.get_4_in3_2_smoothness_index(I[i + 1:i + 2, j:j + 4].T)
                alpha = gamma / (1 + Ii) ** 2
                w = alpha / np.sum(alpha)
                point_eval_approx[i, j] += np.sum(poly_evals_1x4 * w)
        point_eval_approx /= 2

        polynomial_coefs = self.get_polynomial_from4eval_points_and_avg(coords, average_values, point_eval_approx,
                                                                        vertical_segment_eval_approx[2, :],
                                                                        horizontal_segment_eval_approx[:, 2])

        yield PolynomialCell(coords, polynomial_coefs)


class WENO1DPointsRegularCellCreator(CellCreatorBase):
    def __init__(self, num_coeffs=4):
        self.num_coeffs = num_coeffs

    @staticmethod
    def get_4_in3_2_decomposition(avg):
        return np.array([-avg[0] / 2 + 3 * avg[1] / 2, avg[1] / 2 + avg[2] / 2, 3 * avg[2] / 2 - avg[3] / 2])

    @staticmethod
    def get_5_in3_3_decomposition(avg):
        return np.array([23 * avg[0] / 24 - 35 * avg[1] / 12 + 71 * avg[2] / 24,
                         -avg[1] / 24 + avg[2] / 12 + 23 * avg[3] / 24,
                         -avg[2] / 24 + 13 * avg[3] / 12 - avg[4] / 24])

    @staticmethod
    def get_4_in3_2_smoothness_index(smoothness_index, summary=np.max):
        return np.array([summary(smoothness_index[i:i + 1]) for i in range(3)])

    def get_polynomial_from8eval_points_and_avg(self, coords, average_values, eval_points, middle_point_v,
                                                middle_point_h):
        e = eval_points
        x0, y0 = coords.tuple
        avg = average_values[coords.tuple]
        if self.num_coeffs == 4:
            c = np.zeros((2, 2))
            c[0, 0] = e[0, 0] * x0 * y0 + e[0, 0] * x0 + e[0, 0] * y0 + e[0, 0] - e[0, 1] * x0 * y0 - e[0, 1] * y0 - e[
                1, 0] * x0 * y0 - e[1, 0] * x0 + e[1, 1] * x0 * y0
            c[0, 1] = -e[0, 0] * x0 - e[0, 0] + e[0, 1] * x0 + e[0, 1] + e[1, 0] * x0 - e[1, 1] * x0
            c[1, 0] = -e[0, 0] * y0 - e[0, 0] + e[0, 1] * y0 + e[1, 0] * y0 + e[1, 0] - e[1, 1] * y0
            c[1, 1] = e[0, 0] - e[0, 1] - e[1, 0] + e[1, 1]
        elif self.num_coeffs == 5:
            c = np.zeros((3, 3))
            c[0, 0] = -3 * avg * x0 ** 2 - 3 * avg * x0 - 3 * avg * y0 ** 2 - 3 * avg * y0 + 3 * e[0, 0] * x0 ** 2 / 4 + \
                      e[0, 0] * x0 * y0 + 7 * e[0, 0] * x0 / 4 + 3 * e[0, 0] * y0 ** 2 / 4 + 7 * e[0, 0] * y0 / 4 + e[
                          0, 0] + 3 * e[0, 1] * x0 ** 2 / 4 - e[0, 1] * x0 * y0 + 3 * e[0, 1] * x0 / 4 + 3 * e[
                          0, 1] * y0 ** 2 / 4 - e[0, 1] * y0 / 4 + 3 * e[1, 0] * x0 ** 2 / 4 - e[1, 0] * x0 * y0 - e[
                          1, 0] * x0 / 4 + 3 * e[1, 0] * y0 ** 2 / 4 + 3 * e[1, 0] * y0 / 4 + 3 * e[
                          1, 1] * x0 ** 2 / 4 + e[1, 1] * x0 * y0 + 3 * e[1, 1] * x0 / 4 + 3 * e[
                          1, 1] * y0 ** 2 / 4 + 3 * e[1, 1] * y0 / 4
            c[0, 1] = 6 * avg * x0 + 3 * avg - 3 * e[0, 0] * x0 / 2 - e[0, 0] * y0 - 7 * e[0, 0] / 4 - 3 * e[
                0, 1] * x0 / 2 + e[0, 1] * y0 - 3 * e[0, 1] / 4 - 3 * e[1, 0] * x0 / 2 + e[1, 0] * y0 + e[
                          1, 0] / 4 - 3 * e[1, 1] * x0 / 2 - e[1, 1] * y0 - 3 * e[1, 1] / 4
            c[0, 2] = -3 * avg + 3 * e[0, 0] / 4 + 3 * e[0, 1] / 4 + 3 * e[1, 0] / 4 + 3 * e[1, 1] / 4
            c[1, 0] = 6 * avg * y0 + 3 * avg - e[0, 0] * x0 - 3 * e[0, 0] * y0 / 2 - 7 * e[0, 0] / 4 + e[
                0, 1] * x0 - 3 * e[0, 1] * y0 / 2 + e[0, 1] / 4 + e[1, 0] * x0 - 3 * e[1, 0] * y0 / 2 - 3 * e[
                          1, 0] / 4 - e[1, 1] * x0 - 3 * e[1, 1] * y0 / 2 - 3 * e[1, 1] / 4
            c[1, 1] = e[0, 0] - e[0, 1] - e[1, 0] + e[1, 1]
            c[1, 2] = 0
            c[2, 0] = -3 * avg + 3 * e[0, 0] / 4 + 3 * e[0, 1] / 4 + 3 * e[1, 0] / 4 + 3 * e[1, 1] / 4
            c[2, 1] = 0
            c[2, 2] = 0
        elif self.num_coeffs == 9:
            sv = middle_point_v
            sh = middle_point_h
            c = np.zeros((3, 3))
            c[
                0, 0] = 36 * avg * x0 ** 2 * y0 ** 2 + 36 * avg * x0 ** 2 * y0 + 36 * avg * x0 * y0 ** 2 + 36 * avg * x0 * y0 + 3 * \
                        e[0, 0] * x0 ** 2 * y0 ** 2 + 5 * e[0, 0] * x0 ** 2 * y0 + 2 * e[0, 0] * x0 ** 2 + 5 * e[
                            0, 0] * x0 * y0 ** 2 + 8 * e[0, 0] * x0 * y0 + 3 * e[0, 0] * x0 + 2 * e[
                            0, 0] * y0 ** 2 + 3 * e[0, 0] * y0 + e[0, 0] + 3 * e[0, 1] * x0 ** 2 * y0 ** 2 + e[
                            0, 1] * x0 ** 2 * y0 + 5 * e[0, 1] * x0 * y0 ** 2 + 2 * e[0, 1] * x0 * y0 + 2 * e[
                            0, 1] * y0 ** 2 + e[0, 1] * y0 + 3 * e[1, 0] * x0 ** 2 * y0 ** 2 + 5 * e[
                            1, 0] * x0 ** 2 * y0 + 2 * e[1, 0] * x0 ** 2 + e[1, 0] * x0 * y0 ** 2 + 2 * e[
                            1, 0] * x0 * y0 + e[1, 0] * x0 + 3 * e[1, 1] * x0 ** 2 * y0 ** 2 + e[1, 1] * x0 ** 2 * y0 + \
                        e[1, 1] * x0 * y0 ** 2 - 12 * sh[0] * x0 ** 2 * y0 ** 2 - 12 * sh[0] * x0 ** 2 * y0 - 16 * sh[
                            0] * x0 * y0 ** 2 - 16 * sh[0] * x0 * y0 - 4 * sh[0] * y0 ** 2 - 4 * sh[0] * y0 - 12 * sh[
                            1] * x0 ** 2 * y0 ** 2 - 12 * sh[1] * x0 ** 2 * y0 - 8 * sh[1] * x0 * y0 ** 2 - 8 * sh[
                            1] * x0 * y0 - 12 * sv[0] * x0 ** 2 * y0 ** 2 - 16 * sv[0] * x0 ** 2 * y0 - 4 * sv[
                            0] * x0 ** 2 - 12 * sv[0] * x0 * y0 ** 2 - 16 * sv[0] * x0 * y0 - 4 * sv[0] * x0 - 12 * sv[
                            1] * x0 ** 2 * y0 ** 2 - 8 * sv[1] * x0 ** 2 * y0 - 12 * sv[1] * x0 * y0 ** 2 - 8 * sv[
                            1] * x0 * y0
            c[0, 1] = -72 * avg * x0 * y0 ** 2 - 72 * avg * x0 * y0 - 36 * avg * y0 ** 2 - 36 * avg * y0 - 6 * e[
                0, 0] * x0 * y0 ** 2 - 10 * e[0, 0] * x0 * y0 - 4 * e[0, 0] * x0 - 5 * e[0, 0] * y0 ** 2 - 8 * e[
                          0, 0] * y0 - 3 * e[0, 0] - 6 * e[0, 1] * x0 * y0 ** 2 - 2 * e[0, 1] * x0 * y0 - 5 * e[
                          0, 1] * y0 ** 2 - 2 * e[0, 1] * y0 - 6 * e[1, 0] * x0 * y0 ** 2 - 10 * e[1, 0] * x0 * y0 - 4 * \
                      e[1, 0] * x0 - e[1, 0] * y0 ** 2 - 2 * e[1, 0] * y0 - e[1, 0] - 6 * e[1, 1] * x0 * y0 ** 2 - 2 * \
                      e[1, 1] * x0 * y0 - e[1, 1] * y0 ** 2 + 24 * sh[0] * x0 * y0 ** 2 + 24 * sh[0] * x0 * y0 + 16 * \
                      sh[0] * y0 ** 2 + 16 * sh[0] * y0 + 24 * sh[1] * x0 * y0 ** 2 + 24 * sh[1] * x0 * y0 + 8 * sh[
                          1] * y0 ** 2 + 8 * sh[1] * y0 + 24 * sv[0] * x0 * y0 ** 2 + 32 * sv[0] * x0 * y0 + 8 * sv[
                          0] * x0 + 12 * sv[0] * y0 ** 2 + 16 * sv[0] * y0 + 4 * sv[0] + 24 * sv[
                          1] * x0 * y0 ** 2 + 16 * sv[1] * x0 * y0 + 12 * sv[1] * y0 ** 2 + 8 * sv[1] * y0
            c[0, 2] = 36 * avg * y0 ** 2 + 36 * avg * y0 + 3 * e[0, 0] * y0 ** 2 + 5 * e[0, 0] * y0 + 2 * e[0, 0] + 3 * \
                      e[0, 1] * y0 ** 2 + e[0, 1] * y0 + 3 * e[1, 0] * y0 ** 2 + 5 * e[1, 0] * y0 + 2 * e[1, 0] + 3 * e[
                          1, 1] * y0 ** 2 + e[1, 1] * y0 - 12 * sh[0] * y0 ** 2 - 12 * sh[0] * y0 - 12 * sh[
                          1] * y0 ** 2 - 12 * sh[1] * y0 - 12 * sv[0] * y0 ** 2 - 16 * sv[0] * y0 - 4 * sv[0] - 12 * sv[
                          1] * y0 ** 2 - 8 * sv[1] * y0
            c[1, 0] = -72 * avg * x0 ** 2 * y0 - 36 * avg * x0 ** 2 - 72 * avg * x0 * y0 - 36 * avg * x0 - 6 * e[
                0, 0] * x0 ** 2 * y0 - 5 * e[0, 0] * x0 ** 2 - 10 * e[0, 0] * x0 * y0 - 8 * e[0, 0] * x0 - 4 * e[
                          0, 0] * y0 - 3 * e[0, 0] - 6 * e[0, 1] * x0 ** 2 * y0 - e[0, 1] * x0 ** 2 - 10 * e[
                          0, 1] * x0 * y0 - 2 * e[0, 1] * x0 - 4 * e[0, 1] * y0 - e[0, 1] - 6 * e[
                          1, 0] * x0 ** 2 * y0 - 5 * e[1, 0] * x0 ** 2 - 2 * e[1, 0] * x0 * y0 - 2 * e[1, 0] * x0 - 6 * \
                      e[1, 1] * x0 ** 2 * y0 - e[1, 1] * x0 ** 2 - 2 * e[1, 1] * x0 * y0 + 24 * sh[
                          0] * x0 ** 2 * y0 + 12 * sh[0] * x0 ** 2 + 32 * sh[0] * x0 * y0 + 16 * sh[0] * x0 + 8 * sh[
                          0] * y0 + 4 * sh[0] + 24 * sh[1] * x0 ** 2 * y0 + 12 * sh[1] * x0 ** 2 + 16 * sh[
                          1] * x0 * y0 + 8 * sh[1] * x0 + 24 * sv[0] * x0 ** 2 * y0 + 16 * sv[0] * x0 ** 2 + 24 * sv[
                          0] * x0 * y0 + 16 * sv[0] * x0 + 24 * sv[1] * x0 ** 2 * y0 + 8 * sv[1] * x0 ** 2 + 24 * sv[
                          1] * x0 * y0 + 8 * sv[1] * x0
            c[1, 1] = 144 * avg * x0 * y0 + 72 * avg * x0 + 72 * avg * y0 + 36 * avg + 12 * e[0, 0] * x0 * y0 + 10 * e[
                0, 0] * x0 + 10 * e[0, 0] * y0 + 8 * e[0, 0] + 12 * e[0, 1] * x0 * y0 + 2 * e[0, 1] * x0 + 10 * e[
                          0, 1] * y0 + 2 * e[0, 1] + 12 * e[1, 0] * x0 * y0 + 10 * e[1, 0] * x0 + 2 * e[1, 0] * y0 + 2 * \
                      e[1, 0] + 12 * e[1, 1] * x0 * y0 + 2 * e[1, 1] * x0 + 2 * e[1, 1] * y0 - 48 * sh[
                          0] * x0 * y0 - 24 * sh[0] * x0 - 32 * sh[0] * y0 - 16 * sh[0] - 48 * sh[1] * x0 * y0 - 24 * \
                      sh[1] * x0 - 16 * sh[1] * y0 - 8 * sh[1] - 48 * sv[0] * x0 * y0 - 32 * sv[0] * x0 - 24 * sv[
                          0] * y0 - 16 * sv[0] - 48 * sv[1] * x0 * y0 - 16 * sv[1] * x0 - 24 * sv[1] * y0 - 8 * sv[1]
            c[1, 2] = -72 * avg * y0 - 36 * avg - 6 * e[0, 0] * y0 - 5 * e[0, 0] - 6 * e[0, 1] * y0 - e[0, 1] - 6 * e[
                1, 0] * y0 - 5 * e[1, 0] - 6 * e[1, 1] * y0 - e[1, 1] + 24 * sh[0] * y0 + 12 * sh[0] + 24 * sh[
                          1] * y0 + 12 * sh[1] + 24 * sv[0] * y0 + 16 * sv[0] + 24 * sv[1] * y0 + 8 * sv[1]
            c[2, 0] = 36 * avg * x0 ** 2 + 36 * avg * x0 + 3 * e[0, 0] * x0 ** 2 + 5 * e[0, 0] * x0 + 2 * e[0, 0] + 3 * \
                      e[0, 1] * x0 ** 2 + 5 * e[0, 1] * x0 + 2 * e[0, 1] + 3 * e[1, 0] * x0 ** 2 + e[1, 0] * x0 + 3 * e[
                          1, 1] * x0 ** 2 + e[1, 1] * x0 - 12 * sh[0] * x0 ** 2 - 16 * sh[0] * x0 - 4 * sh[0] - 12 * sh[
                          1] * x0 ** 2 - 8 * sh[1] * x0 - 12 * sv[0] * x0 ** 2 - 12 * sv[0] * x0 - 12 * sv[
                          1] * x0 ** 2 - 12 * sv[1] * x0
            c[2, 1] = -72 * avg * x0 - 36 * avg - 6 * e[0, 0] * x0 - 5 * e[0, 0] - 6 * e[0, 1] * x0 - 5 * e[0, 1] - 6 * \
                      e[1, 0] * x0 - e[1, 0] - 6 * e[1, 1] * x0 - e[1, 1] + 24 * sh[0] * x0 + 16 * sh[0] + 24 * sh[
                          1] * x0 + 8 * sh[1] + 24 * sv[0] * x0 + 12 * sv[0] + 24 * sv[1] * x0 + 12 * sv[1]
            c[2, 2] = 36 * avg + 3 * e[0, 0] + 3 * e[0, 1] + 3 * e[1, 0] + 3 * e[1, 1] - 12 * sh[0] - 12 * sh[1] - 12 * \
                      sv[0] - 12 * sv[1]

        else:
            raise Exception(f"Not implemented degree {self.num_coeffs}")
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

        gamma = np.array([1 / 6, 2 / 3, 1 / 6])

        vertical_segment_eval_approx = np.zeros((5, 2))
        for i in range(5):
            for j in range(2):
                poly_evals_1x4 = self.get_4_in3_2_decomposition(avg[i, j:j + 4])
                Ii = self.get_4_in3_2_smoothness_index(I[i, j:j + 4])
                alpha = gamma / (1 + Ii) ** 2
                w = alpha / np.sum(alpha)
                vertical_segment_eval_approx[i, j] = np.sum(poly_evals_1x4 * w)

        horizontal_segment_eval_approx = np.zeros((2, 5))
        for i in range(2):
            for j in range(5):
                poly_evals_1x4 = self.get_4_in3_2_decomposition(avg[i:i + 4, j])
                Ii = self.get_4_in3_2_smoothness_index(I[i:i + 4, j])
                alpha = gamma / (1 + Ii) ** 2
                w = alpha / np.sum(alpha)
                horizontal_segment_eval_approx[i, j] = np.sum(poly_evals_1x4 * w)

        point_eval_approx = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                poly_evals_1x4 = self.get_4_in3_2_decomposition(vertical_segment_eval_approx[i:i + 4, j])
                Ii = self.get_4_in3_2_smoothness_index(I[i:i + 4, j + 1:j + 2])
                alpha = gamma / (1 + Ii) ** 2
                w = alpha / np.sum(alpha)
                point_eval_approx[i, j] = np.sum(poly_evals_1x4 * w)

                poly_evals_1x4 = self.get_4_in3_2_decomposition(horizontal_segment_eval_approx[i, j:j + 4])
                Ii = self.get_4_in3_2_smoothness_index(I[i + 1:i + 2, j:j + 4].T)
                alpha = gamma / (1 + Ii) ** 2
                w = alpha / np.sum(alpha)
                point_eval_approx[i, j] += np.sum(poly_evals_1x4 * w)
        point_eval_approx /= 2

        gamma2 = np.array([9 / 1840, 99 / 920, 71 / 80])
        middle_point_eval_approx_v = np.zeros(2)
        middle_point_eval_approx_h = np.zeros(2)
        for i in range(2):
            poly_evals_1x5 = self.get_5_in3_3_decomposition(vertical_segment_eval_approx[:, i])
            Ii = self.get_4_in3_2_smoothness_index(I[:, i + 1:i + 2])
            alpha = gamma2 / (1 + Ii) ** 2
            w = alpha / np.sum(alpha)
            middle_point_eval_approx_v[i] = np.sum(poly_evals_1x5 * w)

            poly_evals_1x5 = self.get_5_in3_3_decomposition(horizontal_segment_eval_approx[i, :])
            Ii = self.get_4_in3_2_smoothness_index(I[i + 1:i + 2, :].T)
            alpha = gamma2 / (1 + Ii) ** 2
            w = alpha / np.sum(alpha)
            middle_point_eval_approx_h[i] = np.sum(poly_evals_1x5 * w)

        polynomial_coefs = self.get_polynomial_from8eval_points_and_avg(coords, average_values, point_eval_approx,
                                                                        middle_point_eval_approx_v,
                                                                        middle_point_eval_approx_h)

        yield PolynomialCell(coords, polynomial_coefs)


if __name__ == "__main__":
    import sympy as sp


    def to_python(s: str):
        for i in range(3):
            for j in range(3):
                for v in ["c", "e"]:
                    s = s.replace(f"{v}_{i}_{j}", f"{v}[{i}, {j}]")

        for i in range(-2, 5):
            for v in ["sh", "sv", "avg", "c"]:
                s = s.replace(f"{v}_{i}", f"{v}[{i}]")
        return s


    # --------------- --------------- --------------- #
    # WENO 4
    def get_weno_1d_coefs(num_cells, eval_x, sub_polys_num_coefs):
        def create_poly(vector):
            c = sp.Matrix(sp.symarray("c", (len(vector),)))
            return np.sum(np.diag(np.dot(vector, c))), c

        left_ix = -num_cells // 2

        x = sp.symbols("x")
        avg = sp.Matrix(sp.symarray("avg", (num_cells,)))
        monomials = np.array([x ** i for i in range(num_cells)])

        # p3 = p3.subs(sp.solve([
        #     sp.Eq(sp.integrate(p3, (x, -2, -1)), avg[0]),
        #     sp.Eq(sp.integrate(p3, (x, -1, 0)), avg[1]),
        #     sp.Eq(sp.integrate(p3, (x, 0, 1)), avg[2]),
        #     sp.Eq(sp.integrate(p3, (x, 1, 2)), avg[3]),
        # ], c)).subs([(x, 0)])

        c = sp.Matrix(sp.symarray("c", (sub_polys_num_coefs,)))
        num_sub_polys = num_cells - sub_polys_num_coefs + 1
        ps = []
        for i in range(num_sub_polys):
            p, c = create_poly(monomials[:len(c)])
            ps.append(p.subs(sp.solve(
                [sp.Eq(sp.integrate(p, (x, left_ix + i + j, left_ix + i + j + 1)), avg[i + j]) for j in range(len(c))],
                c)).subs([(x, eval_x)]))

        p, c = create_poly(monomials)
        p = p.subs(sp.solve(
            [sp.Eq(sp.integrate(p, (x, left_ix + j, left_ix + j + 1)), avg[j]) for j in range(len(c))],
            c)).subs([(x, eval_x)])

        pps, c = create_poly(np.array(ps))
        solution = sp.solve([sp.Eq((pps - p).subs(a, 0), 0) for a in avg], c)
        print(solution)
        print("np.array([" + ", ".join([to_python(str(k)) for k in ps]) + "])")
        print("np.array([" + ", ".join([to_python(str(sp.simplify(v))) for v in solution.values()]) + "])")
        return ps, solution


    get_weno_1d_coefs(num_cells=4, eval_x=0, sub_polys_num_coefs=2)
    get_weno_1d_coefs(num_cells=5, eval_x=sp.Rational(1, 2), sub_polys_num_coefs=3)
    # get_weno_1d_coefs(num_cells=5, eval_x=sp.Rational(1, 2), sub_polys_num_coefs=2)

    # --------------- --------------- --------------- #
    avg = sp.symbols("avg")
    x, y = sp.symbols("x y")
    x0, y0 = sp.symbols("x0 y0")
    e = sp.Matrix(sp.symarray("e", (2, 2)))
    sv = sp.Matrix(sp.symarray("sv", (2,)))
    sh = sp.Matrix(sp.symarray("sh", (2,)))
    # x_monomials = np.array([1, x - x0, (x - x0) ** 2])
    # y_monomials = np.array([1, y - y0, (y - y0) ** 2])
    x_monomials = np.array([1, x, x ** 2])
    y_monomials = np.array([1, y, y ** 2])

    # --------------- --------------- --------------- #
    # Tensor mode 9 coefficients with 4 then 5 avgs
    c = sp.Matrix(sp.symarray("c", (3, 3)))
    p = c @ x_monomials @ y_monomials

    solution = sp.solve([
        sp.Eq(p.subs([(x, x0), (y, y0)]), e[0, 0]),
        sp.Eq(p.subs([(x, x0 + 1), (y, y0)]), e[1, 0]),
        sp.Eq(p.subs([(x, x0), (y, y0 + 1)]), e[0, 1]),
        sp.Eq(p.subs([(x, x0 + 1), (y, y0 + 1)]), e[1, 1]),

        sp.Eq(p.subs([(x, x0), (y, y0 + sp.Rational(1, 2))]), sh[0]),
        sp.Eq(p.subs([(x, x0 + 1), (y, y0 + sp.Rational(1, 2))]), sh[1]),
        sp.Eq(p.subs([(x, x0 + sp.Rational(1, 2)), (y, y0)]), sv[0]),
        sp.Eq(p.subs([(x, x0 + sp.Rational(1, 2)), (y, y0 + 1)]), sv[1]),

        sp.Eq(sp.integrate(p, (x, x0, x0 + 1), (y, y0, y0 + 1)), avg),
    ], c)

    for k, v in solution.items():
        print(f"{to_python(str(k))} =  {to_python(str(sp.simplify(v)))}")

    # --------------- --------------- --------------- #
    # Tensor mode 9 coefficients
    c = sp.Matrix(sp.symarray("c", (3, 3)))
    p = c @ x_monomials @ y_monomials

    solution = sp.solve([
        sp.Eq(p.subs([(x, x0), (y, y0)]), e[0, 0]),
        sp.Eq(p.subs([(x, x0 + 1), (y, y0)]), e[1, 0]),
        sp.Eq(p.subs([(x, x0), (y, y0 + 1)]), e[0, 1]),
        sp.Eq(p.subs([(x, x0 + 1), (y, y0 + 1)]), e[1, 1]),

        sp.Eq(sp.integrate(p, (x, x0, x0 + 1)).subs([(y, y0)]), sv[0]),
        sp.Eq(sp.integrate(p, (x, x0, x0 + 1)).subs([(y, y0 + 1)]), sv[1]),
        sp.Eq(sp.integrate(p, (y, y0, y0 + 1)).subs([(x, x0)]), sh[0]),
        sp.Eq(sp.integrate(p, (y, y0, y0 + 1)).subs([(x, x0 + 1)]), sh[1]),

        sp.Eq(sp.integrate(p, (x, x0, x0 + 1), (y, y0, y0 + 1)), avg),
    ], c)

    for k, v in solution.items():
        print(f"{to_python(str(k))} =  {to_python(str(sp.simplify(v)))}")

    # --------------- --------------- --------------- #
    # Tensor mode 4 coefficients
    c = sp.Matrix(sp.symarray("c", (2, 2)))
    p = c @ x_monomials[:2] @ y_monomials[:2]

    solution = sp.solve([
        sp.Eq(p.subs([(x, x0), (y, y0)]), e[0, 0]),
        sp.Eq(p.subs([(x, x0 + 1), (y, y0)]), e[1, 0]),
        sp.Eq(p.subs([(x, x0), (y, y0 + 1)]), e[0, 1]),
        sp.Eq(p.subs([(x, x0 + 1), (y, y0 + 1)]), e[1, 1]),
    ], c)

    for k, v in solution.items():
        print(f"{to_python(str(k))} =  {to_python(str(sp.simplify(v)))}")

    # --------------- --------------- --------------- #
    # Tensor mode 5 coefficients
    c = sp.Matrix(sp.symarray("c", (3, 3)))
    p = c @ x_monomials @ y_monomials

    solution = sp.solve([
        sp.Eq(p.subs([(x, x0), (y, y0)]), e[0, 0]),
        sp.Eq(p.subs([(x, x0 + 1), (y, y0)]), e[1, 0]),
        sp.Eq(p.subs([(x, x0), (y, y0 + 1)]), e[0, 1]),
        sp.Eq(p.subs([(x, x0 + 1), (y, y0 + 1)]), e[1, 1]),

        sp.Eq(sp.integrate(p, (x, x0, x0 + 1), (y, y0, y0 + 1)), avg),
        sp.Eq(c[0, 2], c[2, 0]),

        sp.Eq(c[1, 2], 0),
        sp.Eq(c[2, 1], 0),
        sp.Eq(c[2, 2], 0),
    ], c)

    for k, v in solution.items():
        print(f"{to_python(str(k))} =  {to_python(str(sp.simplify(v)))}")
