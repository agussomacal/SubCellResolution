from collections import namedtuple
from typing import Union, List

import numpy as np

from lib.Curves.Curves import Curve

TrigoParams = namedtuple("TrigoParams", "x0 y0 amplitude frequency")

import sympy as sp

# Define symbols
x = sp.Symbol('x', real=True)
y0 = sp.Symbol('y0', real=True)
x0 = sp.Symbol('x0', real=True)
amplitude = sp.Symbol('amplitude', real=True)
freq = sp.Symbol('freq', positive=True)  # freq > 0 for valid division

# Define the symbolic integral
omega = 2 * sp.pi * freq  # Angular frequency

integral_expr = (y0 * (x - x0)
                 - (amplitude / omega)
                 * (sp.cos(omega * (x - x0)) - 1))

integral_expr = integral_expr.simplify()
integral_expression_lambdified = sp.lambdify([x, y0, x0, amplitude, freq], integral_expr, 'numpy')


def get_concavity(x_points, y):
    return len(x_points) * (2 * (np.diff(y, 2).squeeze() > 0) - 1)


class CurveTrigo(Curve):
    def __init__(self, params: TrigoParams, value_up=0, value_down=1):
        """

        :param params:
        :param value_up:
        :param value_down:
        :param concave: True if down semi-circle or positive curvature. False otherwise.
        """
        super().__init__(value_up=value_up, value_down=value_down)
        self.x0, self.y0, self.amplitude, self.freq = params

    @property
    def params(self):
        return self.x0, self.y0, self.amplitude, self.freq

    @params.setter
    def params(self, args):
        self.x0, self.y0, self.amplitude, self.freq = args

    def set_x_shift(self, shift):
        self.x0 += shift

    def set_y_shift(self, shift):
        self.y0 += shift

    def function(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.y0 + np.sin(2 * np.pi * self.freq * (x - self.x0)) * self.amplitude

    def function_inverse(self, y: float) -> List[float]:
        main_solution = np.arcsin((y - self.y0) / self.amplitude) / (2 * np.pi * self.freq)
        breakpoints = ([self.x0 + main_solution] +
                       [self.x0 + main_solution * (-1) ** k + k / 2 / self.freq for k in [-2, -1, 1, 2]])
        return breakpoints

    def function_integral(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # return (self.y0 * (x - self.x0)
        #         - (self.amplitude / (2 * np.pi * self.freq))
        #         * (np.cos(2 * np.pi * self.freq * (x - self.x0))))
        return integral_expression_lambdified(x, self.y0, self.x0, self.amplitude, self.freq)
