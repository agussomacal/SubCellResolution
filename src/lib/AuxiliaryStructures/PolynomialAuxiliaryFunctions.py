import itertools
from typing import List, Union
from typing import Tuple

import numpy as np
from numpy.polynomial.polynomial import polyval, polyval2d, polyval3d
from sklearn.linear_model import LinearRegression


def evaluate_polynomial(polynomial_coefs: np.ndarray, query_points: Union[np.ndarray, List[Tuple]]):
    """

    :param query_points: list of point coords [(x1, y1, z1), (x2, y2 ... zn)] or ndarray form but m rows with m
    num points
    :return:
    """
    dimensionality = len(np.shape(polynomial_coefs))
    return {1: polyval, 2: polyval2d, 3: polyval3d}[dimensionality](*np.transpose(query_points), polynomial_coefs)


def monomials_integral_in_rectangles_iterator(rectangles: List[np.ndarray], degrees: Tuple):
    """
    :param rectangles: each rectangl has 2 rows with the minimum and maximum values of the hypercube:
    [(x0, y0 ..., k0), (xf, yf ..., kf)]
    :return: for each rectangle the area integrated of the polynomial.
    """
    ndims = len(degrees)
    assert ndims <= 3, \
        "it doesn't work in more dimensions, but it can if you want. Add more letters to einsum"
    max_degree = np.max(degrees)
    monomials_matrix = np.diag(np.polyint(np.ones(max_degree + 1)))[-2::-1, :]

    for rectangle in rectangles:
        # create canonical basis for each dimension and evaluate in the limits
        power_eval = [np.diff(np.array([np.polyval(m, [pos_min, pos_max])
                                        for m in monomials_matrix[:degrees[dim] + 1]]).T, axis=0).ravel()
                      for dim, (pos_min, pos_max) in enumerate(zip(*rectangle))]

        yield power_eval


def evaluate_polynomial_integral_in_rectangle(poly_coefs: np.ndarray, rectangles: List[np.ndarray]):
    """

    :param poly_coefs: it is a tensor with the coefficients of the polynomial arranged so that the constant is the
    poly_coefs[0 ,0,...,0]

    :param rectangles: each rectangl has 2 rows with the minimum and maximum values of the hypercube:
    [(x0, y0 ..., k0), (xf, yf ..., kf)]
    :return: for each rectangle the area integrated of the polynomial.
    """
    degrees = np.array(np.shape(poly_coefs)) - 1
    ndims = len(degrees)
    integrals = [
        np.einsum("i,j,k,"[:ndims * 2] + "ijk"[:ndims],
                  *monomials_integral_eval, poly_coefs)
        for monomials_integral_eval in monomials_integral_in_rectangles_iterator(rectangles, degrees)
    ]
    return integrals


def dimensions_iterator(ndims: int, degree: int) -> Tuple[int]:
    # make cartesian product of degree-coefs and then filters only those that are below the degree value.
    for coef_ix in itertools.product(*[np.arange(degree + 1) for _ in range(ndims)]):
        # (1, 1, 1); (x, 1, 1); (x**2, 1, 1); (x, y, 1); (x, 1, z) but not (x**2, y, 1)
        if np.sum(coef_ix) <= degree:
            yield tuple(coef_ix)


def fit_polynomial_from_integrals(rectangles, values, degree: int, sample_weight=None):
    """

    :param stencil:
        * In 1d are the x values where the cell starts, then in the loop we add +1.
        * In 2d is the same, but now we add in each dimension +1 so the integral is from stencil[i] to stencil[i]+1

    :param values: the value of the integral in each cell of the stencil.
    :return:
    """
    ndims = len(rectangles[0][0])
    polynomial_coefs = np.zeros(np.repeat(degree + 1, ndims))
    dims_ix = tuple(range(ndims))  # (0, 1, .. self.ndims)

    A = []
    for monomials_integrals in monomials_integral_in_rectangles_iterator(rectangles, tuple([degree] * ndims)):
        A.append(
            [np.prod(np.array(monomials_integrals)[dims_ix, coef_ix]) for coef_ix in
             dimensions_iterator(ndims=ndims, degree=degree)])

    coefs = LinearRegression(fit_intercept=False).fit(A, values, sample_weight=sample_weight).coef_

    for i, coef_ix in enumerate(dimensions_iterator(ndims=ndims, degree=degree)):
        polynomial_coefs[coef_ix] = coefs[i]

    return polynomial_coefs
