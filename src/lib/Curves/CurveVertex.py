import operator
import warnings
from typing import Union, List, Tuple

import numpy as np
from numpy.polynomial import Polynomial

from lib.Curves.CurvePolynomial import LEFT, RIGHT, CurvePolynomialByParts, NoCurveRegion
from lib.Curves.Curves import CurvesCombined


def create_curve_vertex_polynomial(polynomials: Tuple[Union[List, np.ndarray, Polynomial], ...], x0: float,
                                   value_up: float = 0, value_down: float = 1, x_shift: float = 0,
                                   directions=(LEFT, RIGHT)) -> Tuple[
    Union[CurvePolynomialByParts, NoCurveRegion], ...]:
    # this method only works if functions are 0-1
    assert np.allclose(value_up, 1) or np.allclose(value_up, 0)
    if not np.allclose(np.abs(value_up - value_down), 1):
        warnings.warn(f"value_up - value_down: {np.abs(value_up - value_down)}")

    x0 = x0 + x_shift
    # Both half curves depart from the vertex to the same direction
    if directions[0] == directions[1]:
        # Create each polynomial
        polynomial0 = polynomials[0] if isinstance(polynomials[0], Polynomial) else Polynomial(polynomials[0])
        polynomial1 = polynomials[1] if isinstance(polynomials[1], Polynomial) else Polynomial(polynomials[1])
        # which polynomial is above which
        up_down_shift = int(polynomial0(x0 + directions[0] - x_shift) >
                            polynomial1(x0 + directions[1] - x_shift))
        up_down_shift = (up_down_shift == value_up) * 1
        # Create polynomial curves
        polynomial_1 = CurvePolynomialByParts(
            polynomial=polynomials[0], x_shift=x_shift,
            value_up=value_up, value_down=value_down,
            x0=x0, direction=directions[0])
        polynomial_2 = CurvePolynomialByParts(
            polynomial=polynomials[1], x_shift=x_shift,
            value_up=up_down_shift - value_up, value_down=up_down_shift - value_down,
            x0=x0, direction=directions[1])
        # Check the value in between curves, this should be the opposite in the NoRegion side
        xt = x0 + directions[0]
        polynomial = polynomial_1 + polynomial_2
        return polynomial_1, polynomial_2, NoCurveRegion(value=1 - polynomial(xt, np.mean(polynomial(np.array([xt])))),
                                                         x0=x0, direction=-1 * directions[0])
    else:
        polynomial_1 = CurvePolynomialByParts(
            polynomial=polynomials[0], x_shift=x_shift,
            value_up=value_up, value_down=value_down,
            x0=x0, direction=directions[0])
        polynomial_2 = CurvePolynomialByParts(
            polynomial=polynomials[1], x_shift=x_shift,
            value_up=value_up, value_down=value_down,
            x0=x0, direction=directions[1])

        return polynomial_1, polynomial_2


class CurveVertexPolynomial(CurvesCombined):
    def __init__(self, polynomials: Tuple[Union[List, np.ndarray, Polynomial], ...], x0: float, value_up: float = 0,
                 value_down: float = 1, x_shift: float = 0, directions=(LEFT, RIGHT)):
        # TODO: when LEFT RIGHT branches are evaluated exactly in x0 both are summed.
        curves = create_curve_vertex_polynomial(polynomials, x0, value_up, value_down, x_shift, directions)
        super(CurveVertexPolynomial, self).__init__(operator.add, *curves)
        self.curve_name = "Vertex" + self.curve_name

    @property
    def x0(self):
        return self.curves[0].x0

    @x0.setter
    def x0(self, value):
        for i in range(self.num_curves):
            self.curves[i].x0 = value

    @property
    def params(self):
        params = super(CurveVertexPolynomial, self).params
        # [parameters of first polynomial] [parameters of second polynomial except y0 because is included in first] [x0]
        return np.hstack((params[:self.curves[0].dim], params[self.curves[0].dim + 1:]))

    @params.setter
    def params(self, args):
        self.x0 = args[-1]
        # [parameters of first polynomial] [] [x0]
        params = np.hstack((args[:self.curves[0].dim], args[:1], args[self.curves[0].dim:]))
        super(CurveVertexPolynomial, self.__class__).params.fset(self, params)


class CurveVertexLinearAngle(CurveVertexPolynomial):
    def __init__(self, angle1, angle2, x0: float, y0: float, value_up, value_down):
        """
        angle1: first line angle
        angle2: second line angle respect to first
        """
        slope1 = np.tan(angle1)
        slope2 = np.tan(angle1 + angle2)
        super().__init__(polynomials=([y0, slope1], [y0, slope2]),
                         x0=0, value_up=value_up, value_down=value_down, x_shift=x0,
                         directions=(int(np.sign(np.cos(angle1))), int(np.sign(np.cos(angle1 + angle2)))))
