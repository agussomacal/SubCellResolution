import numpy as np

from lib.Curves.Curves import CurveReparametrized
from lib.Curves.CurvePolynomial import CurvePolynomial


class CurveAvg(CurveReparametrized):
    @property
    def params(self):
        """
        :return: The average of the 1d-cells
        """
        return tuple([self.function_integral(x + 0.5) - self.function_integral(x - 0.5) for x in self.x_points])


class CurveAveragePolynomial(CurveReparametrized, CurvePolynomial):
    def __init__(self, x_points, y_points, value_up=0, value_down=1, degree=1, ccew=0, center=None):
        """

        :param x_points:
        :param y_points:
        :param value_up:
        :param value_down:
        :param degree:
        :param ccew: central cell extra weight
        """
        self.degree = degree
        super().__init__(x_points, y_points, value_down=value_down, value_up=value_up, ccew=ccew, center=center)

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


if __name__ == "__main__":
    degree = 2
    x_points = np.arange(3) + 0.5
    A = (np.vander(x_points + 0.5, N=degree + 2, increasing=True)[:, 1:] -
         np.vander(x_points - 0.5, N=degree + 2, increasing=True)[:, 1:]) / np.arange(1, degree + 2)[np.newaxis, :]
    print(A)
    # np.linalg.lstsq(
    #     A  * weights[:, np.newaxis],
    #     (y_points * weights).reshape((-1, 1)),
    #     rcond=None)[0].ravel()

    cap = CurveAveragePolynomial(x_points=[0, 1, 2], y_points=[0, 1, 4], value_up=0, value_down=1, degree=2, ccew=0,
                           center=None)
    print(cap.x_points, cap.params, super(CurveAveragePolynomial, cap).params)
    cap.set_x_shift(1)
    cap.set_y_shift(10)
    print(cap.x_points, cap.params, super(CurveAveragePolynomial, cap).params)
