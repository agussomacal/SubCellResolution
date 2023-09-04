from collections import namedtuple
from typing import Union, List

import numpy as np

from lib.Curves.Curves import Curve

CircleParams = namedtuple("CircleParams", "x0 y0 radius")


def get_concavity(x_points, y):
    return len(x_points) * (2 * (np.diff(y, 2).squeeze() > 0) - 1)


class CurveSemiCircle(Curve):
    # TODO: what happens if circle goes below... Check
    def __init__(self, params: CircleParams, value_up=0, value_down=1, concave=False):
        """

        :param params:
        :param value_up:
        :param value_down:
        :param concave: True if down semi-circle or positive curvature. False otherwise.
        """
        super().__init__(value_up=value_up, value_down=value_down)
        self.x0, self.y0, self.r = params
        self.concave = concave

    @property
    def params(self):
        return self.x0, self.y0, self.r

    @params.setter
    def params(self, args):
        self.x0, self.y0, self.r = args

    def set_x_shift(self, shift):
        self.x0 += shift

    def set_y_shift(self, shift):
        self.y0 += shift

    def function(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        dy = np.sqrt(self.r ** 2 - (x - self.x0) ** 2)
        if self.concave:
            return self.y0 - dy
        return self.y0 + dy

    def function_inverse(self, y: float) -> List[float]:
        breakpoints = [self.x0 - self.r, self.x0 + self.r]
        if (y > self.y0 and not self.concave) or (y < self.y0 and self.concave):
            discriminantish = np.sqrt(self.r ** 2 - (y - self.y0) ** 2)
            if not np.isnan(discriminantish):
                breakpoints += [self.x0 - discriminantish, self.x0 + discriminantish]
        return breakpoints

    def function_integral(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        res = np.zeros(np.shape(x))
        beyond_circle = x > self.x0 + self.r
        res[beyond_circle] = np.pi * self.r ** 2 / 2

        in_circle = (self.x0 + self.r >= x) & (x >= self.x0 - self.r)
        dx = self.x0 - x[in_circle]
        alpha = np.arccos(dx / self.r)
        res[in_circle] = alpha * self.r ** 2 / 2 - dx * np.sin(alpha) * self.r / 2

        res *= (-1) ** self.concave

        res[in_circle] += self.y0 * (x[in_circle] - (self.x0 - self.r))
        res[beyond_circle] += self.y0 * 2 * self.r
        return res


class CurveCircle(CurveSemiCircle):
    # TODO: what happens if circle goes below... Check
    def __init__(self, params: CircleParams, value_up=0, value_down=1):
        # super().__init__(params=params, value_up=value_up, value_down=value_down)
        super().__init__(params, value_up, value_down)
        new_curve = CurveSemiCircle(params=params, value_down=value_down, value_up=value_up) - \
                    CurveSemiCircle(params=params, value_down=value_down, value_up=value_up, concave=True)
        setattr(self, "calculate_integrals", new_curve.calculate_integrals)
        setattr(self, "function_inverse", new_curve.function_inverse)
        setattr(self, "function", new_curve.function)
        setattr(self, "evaluate", new_curve.evaluate)
