from functools import partial
from typing import Dict, Generator, Tuple, Callable, Type

import numpy as np

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import ArrayIndexerNd
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.CellCreators.CellCreatorBase import CellBase
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesCurveCellCreator
from lib.Curves.AverageCurves import CurveAveragePolynomial
from lib.Curves.CurveCircle import CurveSemiCircle, CircleParams
from lib.Curves.CurvePolynomial import CurvePolynomial
from lib.Curves.Curves import Curve
from lib.Curves.Curves import CurveReparametrized
from lib.StencilCreators import Stencil


class TaylorCurveCellCreator(ValuesCurveCellCreator):
    def __init__(self, curve: Type[Curve], degree, regular_opposite_cell_searcher: Callable, ccew=0):
        super().__init__(
            vander_curve=partial(CurveAveragePolynomial, degree=degree, ccew=ccew),
            regular_opposite_cell_searcher=regular_opposite_cell_searcher, natural_params=False)
        self.curve = curve

    def get_curve_from_taylor(self, curve_polynomial: CurveReparametrized) -> Curve:
        raise Exception("Not implemented.")

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[Curve, None, None]:
        for curve_polynomial in super(TaylorCurveCellCreator, self).create_curves(
                average_values=average_values, indexer=indexer, cells=cells, coords=coords,
                smoothness_index=smoothness_index, independent_axis=independent_axis, stencil=stencil,
                regular_opposite_cells=regular_opposite_cells):
            yield self.get_curve_from_taylor(curve_polynomial)


class TaylorFromVanderCurveCellCreator(TaylorCurveCellCreator):
    """
    Uses a quadratic fit (AERO) to then infer evaluation points and values to estimate the final curve prameters.
    """
    def get_curve_from_taylor(self, curve_polynomial: CurveReparametrized) -> CurveReparametrized:
        xc = curve_polynomial.x_points[curve_polynomial.center]
        x_points = np.linspace(xc - 0.5, xc + 0.5, num=curve_polynomial.dim)
        return self.vander_curve(
            x_points=x_points,
            y_points=curve_polynomial.function(x_points),
            value_up=curve_polynomial.value_up,
            value_down=curve_polynomial.value_down,
            center=curve_polynomial.center
        )


class TaylorCircleCurveCellCreator(TaylorCurveCellCreator):
    """
    For the case of the circle uses the quadratic to get the curvature and then the radius and positions of the center.
    """
    def __init__(self, regular_opposite_cell_searcher: Callable, ccew=0):
        super().__init__(curve=CurveSemiCircle, degree=2, regular_opposite_cell_searcher=regular_opposite_cell_searcher,
                         ccew=ccew)

    def get_curve_from_taylor(self, curve_polynomial: (CurvePolynomial, CurveReparametrized)) -> CurveReparametrized:
        xc = curve_polynomial.x_points[curve_polynomial.center]
        poly = curve_polynomial.polynomial
        c, b, a = poly.coef
        p0 = poly(xc)
        p1 = poly.deriv()(xc)
        p2 = poly.deriv(2)(xc)

        r = np.abs((1 + (2 * a * xc + b) ** 2) ** (3 / 2) / (2 * a))
        dx = r / np.sqrt(1 + p1 ** 2)
        dy = np.sqrt(r ** 2 - dx ** 2)

        # np.array([xc, poly(xc)]) + np.array([-1 / p1, 1]) * r / np.sqrt(1 + 1 / p1 ** 2)

        # p1 = poly.deriv()(xc)
        # 1 / (2 * p2 / (1 + (p1) ** 2) ** (3 / 2)) / 28
        concavity = np.sign(p2)
        return CurveSemiCircle(
            # CircleParams(x0=xc + p1 / p2, y0=p0 - 1 / p2, radius=np.sqrt(1 + p1 ** 2) / p2),
            CircleParams(x0=xc - np.sign(p1) * concavity * dy,
                         y0=p0 + concavity * dx, radius=r),
            value_up=curve_polynomial.value_up,
            value_down=curve_polynomial.value_down,
            concave=p2 > 0,
        )
