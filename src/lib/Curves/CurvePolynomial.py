import operator
from typing import Union, List, Tuple

import numpy as np
from numpy.polynomial import Polynomial

from lib.Curves.Curves import Curve

LEFT = -1
RIGHT = 1
LEFT_RIGHT_OPERATORS = {LEFT: operator.le, RIGHT: operator.ge}


class CurvePolynomial(Curve):
    def __init__(self, polynomial: Union[List, np.ndarray, Polynomial], value_up=0, value_down=1, x_shift=0):
        """
        coef : array_like
        Polynomial coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` give ``1 + 2*x + 3*x**2``.
        """
        self.polynomial = polynomial if isinstance(polynomial, Polynomial) else Polynomial(polynomial)
        self.poly_integral = self.polynomial.integ()
        self.x_shift = x_shift

        if self.polynomial.degree() == 1:
            curve_name = "Line"
        elif self.polynomial.degree() == 2:
            curve_name = "Quadratic"
        else:
            curve_name = "Poly{}".format(self.polynomial.degree())
        super().__init__(curve_name, value_up, value_down)

    @property
    def params(self):
        return self.polynomial.coef

    @params.setter
    def params(self, args):
        self.polynomial = args if isinstance(args, Polynomial) else Polynomial(args)
        self.poly_integral = self.polynomial.integ()

    def set_x_shift(self, shift):
        self.x_shift += shift

    def set_y_shift(self, shift):
        params = np.array(CurvePolynomial.params.fget(self))
        params[0] += shift
        CurvePolynomial.params.fset(self, params)

    def function(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.polynomial(x - self.x_shift)

    def function_inverse(self, y: float) -> List[float]:
        roots = (self.polynomial - y).roots() + self.x_shift
        return roots[np.isreal(roots)]

    def function_integral(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.poly_integral(x - self.x_shift)


class CurvePolynomialByParts(CurvePolynomial):
    def __init__(self, x0: float, direction, polynomial: Union[List, np.ndarray, Polynomial], value_up=0, value_down=1,
                 x_shift=0):
        """
        coef : array_like
        Polynomial coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` give ``1 + 2*x + 3*x**2``.
        """
        self.x0 = x0
        self.direction = direction
        self.direction_op = LEFT_RIGHT_OPERATORS[direction]
        super().__init__(polynomial=polynomial, value_up=value_up, value_down=value_down, x_shift=x_shift)
        self.x0_integral = super(CurvePolynomialByParts, self).function_integral(self.x0)

    def set_x_shift(self, shift):
        self.x_shift += shift
        self.x0 += shift
        self.x0_integral = super(CurvePolynomialByParts, self).function_integral(self.x0)

    def function(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        f = np.array(super(CurvePolynomialByParts, self).function(x))
        f[~self.direction_op(x, self.x0)] = np.nan  # because there is no definition of the function there
        return f

    def function_inverse(self, y: float) -> List[float]:
        roots = np.append(super(CurvePolynomialByParts, self).function_inverse(y), self.x0)
        return roots[self.direction_op(roots, self.x0)]

    def function_integral(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        integral = super(CurvePolynomialByParts, self).function_integral(x)
        if self.direction == RIGHT:
            integral -= self.x0_integral
            integral[x < self.x0] = np.nan  # because there is n definition of the function there
        else:
            integral[x > self.x0] = np.nan  # because there is n definition of the function there
        return integral


def angle_2slope(y0, angle):
    return y0, np.tan(angle)


class CurveLinearAngle(CurvePolynomial):
    def __init__(self, angle, y0, value_up=0, value_down=1, x_shift=0):
        super().__init__(list(angle_2slope(y0, angle)), value_up, value_down, x_shift)

    @property
    def params(self):
        y, slope = CurvePolynomial.params.fget(self)
        return y, np.arctan(slope)

    @params.setter
    def params(self, args):
        CurvePolynomial.params.fset(self, angle_2slope(args[0], args[1]))


class CurveLinearAngleOffset(Curve):
    def __init__(self, angle, r, x0, y0, value_up=0, value_down=1):
        self.y0 = y0
        self.x0 = x0
        self.angle = angle
        self.r = r
        super().__init__(value_up=value_up, value_down=value_down)

    @property
    def params(self):
        return self.r, self.angle

    @params.setter
    def params(self, args):
        self.r = args[0]
        self.angle = args[1]

    @property
    def true_x0(self):
        return self.x0 + self.r * np.cos(self.angle)

    @property
    def true_y0(self):
        return self.y0 + self.r * np.sin(self.angle)

    @property
    def true_slope(self):
        return -1 / np.tan(self.angle)

    def function(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.true_y0 + self.true_slope * (x - self.true_x0)

    def function_inverse(self, y: float) -> List[float]:
        root = self.true_x0 - (y - self.true_y0) * np.tan(self.angle)
        return [] if np.isinf(root) else [root]

    def function_integral(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.true_y0 * (x - self.true_x0) + self.true_slope * (x - self.true_x0) ** 2 / 2


class CurveQuadraticAngle(CurvePolynomial):
    def __init__(self, angle, y0, a, value_up=0, value_down=1, x_shift=0):
        super().__init__([y0, np.tan(angle), a], value_up, value_down, x_shift)


class CurveQuadratic(CurvePolynomial):
    def __init__(self, c, b, a, value_up=0, value_down=1, x_shift=0):
        super().__init__([c, b, a], value_up, value_down, x_shift)


class NoCurveRegion(Curve):
    def __init__(self, value, x0: float, direction):
        super().__init__(curve_name="NoCurveRegion", value_up=value)
        self.x0 = x0
        self.direction = direction
        self.direction_op = LEFT_RIGHT_OPERATORS[direction]

    @property
    def value(self):
        return self.value_up

    @property
    def params(self):
        return [self.x0]

    @params.setter
    def params(self, *args):
        self.x0 = args[0]

    def set_x_shift(self, shift):
        self.x0 += shift

    def set_y_shift(self, shift):
        pass

    def function(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return x * np.nan

    def function_inverse(self, y: float) -> List[float]:
        return [self.x0]

    def function_integral(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return x * np.nan

    def evaluate(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray] = None):
        f = x * 0 + self.value
        f[~self.direction_op(x, self.x0)] = 0  # because there is n definition of the function there
        return f

    def calculate_integrals(self, x_breakpoints, y_limits: Tuple[float, ...]) -> np.ndarray:
        dy = y_limits[1] - y_limits[0]
        x0 = np.reshape(x_breakpoints[:-1], (-1, 1))
        xf = np.reshape(x_breakpoints[1:], (-1, 1))
        dx = (xf - x0)
        return dx * dy * self.evaluate((xf + x0) / 2)
