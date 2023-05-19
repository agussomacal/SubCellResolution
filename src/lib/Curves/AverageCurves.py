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
    def __init__(self, x_points, y_points, value_up=0, value_down=1, degree=1):
        self.degree = degree
        super().__init__(x_points, y_points, value_down=value_down, value_up=value_up)

    def new_params2natural_params(self, x_points, y_points):
        return np.linalg.lstsq(
            (np.vander(x_points + 0.5, N=self.degree + 2, increasing=True)[:, 1:] -
             np.vander(x_points - 0.5, N=self.degree + 2, increasing=True)[:, 1:]) /
            np.arange(1, self.degree + 2)[np.newaxis, :],
            y_points.reshape((-1, 1)),
            rcond=None)[0].ravel()
