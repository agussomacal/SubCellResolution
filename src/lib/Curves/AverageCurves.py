import numpy as np

from lib.Curves.CurveBase import CurveReparametrized
from lib.Curves.CurvePolynomial import CurvePolynomial


class CurveAvg(CurveReparametrized):
    @property
    def params(self):
        """
        :return: The average of the 1d-cells
        """
        return tuple([self.function_integral(x + 0.5) - self.function_integral(x - 0.5) for x in self.x_points])


class CurveAveragePolynomial(CurveReparametrized, CurvePolynomial):
    def __init__(self, x_points, y_points, value_up=0, value_down=1, degree=1, ccew=0):
        """

        :param x_points:
        :param y_points:
        :param value_up:
        :param value_down:
        :param degree:
        :param ccew: central cell extra weight
        """
        self.degree = degree
        self.ccew = ccew
        self.weights = np.ones(len(x_points))
        self.weights[len(x_points) // 2] += self.ccew
        self.weights = np.sqrt(self.weights)
        super().__init__(x_points, y_points, value_down=value_down, value_up=value_up)

    def new_params2natural_params(self, x_points, y_points):
        return np.linalg.lstsq(
            (np.vander(x_points + 0.5, N=self.degree + 2, increasing=True)[:, 1:] -
             np.vander(x_points - 0.5, N=self.degree + 2, increasing=True)[:, 1:]) /
            np.arange(1, self.degree + 2)[np.newaxis, :] * self.weights[:, np.newaxis],
            (y_points * self.weights).reshape((-1, 1)),
            rcond=None)[0].ravel()

    def get_natural_parametrization_curve(self):
        return CurvePolynomial(self.new_params2natural_params(self.x_points, self.y_points), value_up=self.value_up,
                               value_down=self.value_down, x_shift=self.x_shift)
