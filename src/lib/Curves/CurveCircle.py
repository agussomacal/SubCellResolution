import operator
from typing import Union, List, Tuple

import numpy as np
from numpy.polynomial import Polynomial

from lib.Curves.CurveBase import CurveBase


class CurveCircle(CurveBase):
    # TODO: what happens if circle goes below... Check
    def __init__(self, x0: float, y0: float, radius: float, value_up=0, value_down=1):
        super().__init__(value_up=value_up, value_down=value_down)
        self.x0 = x0
        self.y0 = y0
        self.r = radius

    @property
    def params(self):
        return self.x0, self.y0, self.r

    @params.setter
    def params(self, args):
        self.x0, self.y0, self.r = args

    def function(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        dy = np.sqrt(self.r ** 2 - (x - self.x0) ** 2)
        return np.array([self.y0 - dy, self.y0 + dy])

    def function_inverse(self, y: float) -> List[float]:
        discriminantish = np.sqrt(self.r ** 2 - (y - self.y0) ** 2)
        if np.isnan(discriminantish):
            return []
        else:
            return [self.x0 - discriminantish, self.x0 + discriminantish]

    def function_integral(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        res = np.zeros(np.shape(x))
        beyond_circle = x > self.x0 + self.r
        res[beyond_circle] = np.pi * self.r ** 2

        in_circle = (self.x0 + self.r >= x) & (x >= self.x0 - self.r)
        dx = self.x0 - x[in_circle]
        alpha = np.arccos(dx / self.r)
        res[in_circle] = alpha * self.r ** 2
        return res


def points2circle(a, b, c):
    a, b, c = list(map(np.array, [a, b, c]))
    x0, y0 = np.linalg.lstsq([b - a,
                              c - a,
                              c - b],
                             [(a + b) / 2 @ (b - a),
                              (a + c) / 2 @ (c - a),
                              (c + b) / 2 @ (c - b)],
                             rcond=None)[0]
    return x0, y0, \
        (np.sqrt((a[0] - x0) ** 2 + (a[1] - y0) ** 2) +
         np.sqrt((b[0] - x0) ** 2 + (b[1] - y0) ** 2) +
         np.sqrt((c[0] - x0) ** 2 + (c[1] - y0) ** 2)) / 3


class CurveCirclePoints(CurveCircle):
    # TODO: what happens if circle goes below... Check
    def __init__(self, ym: float, yc: float, yp: float, value_up=0, value_down=1, x_shift=0):
        self.ym = ym
        self.yc = yc
        self.yp = yp
        self.x_shift = x_shift
        super().__init__(*points2circle((x_shift - 1, ym), (x_shift, yc), (x_shift + 1, yp)),
                         value_up=value_up, value_down=value_down)

    @property
    def params(self):
        ym = self.function(self.x_shift - 1)
        yc = self.function(self.x_shift)
        yp = self.function(self.x_shift + 1)
        return ym[np.argmin(np.abs(ym - self.ym))], yc[np.argmin(np.abs(yc - self.yc))], yp[
            np.argmin(np.abs(yp - self.yp))]

    @params.setter
    def params(self, args):
        CurveCircle.params.fset(self, points2circle((self.x_shift - 1, args[0]), (self.x_shift, args[1]),
                                                    (self.x_shift + 1, args[2])))
