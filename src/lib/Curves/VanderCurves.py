import numpy as np

from lib.Curves.CurveBase import CurveReparametrized
from lib.Curves.CurveCircle import CurveCircle
from lib.Curves.CurvePolynomial import CurvePolynomial


class CurveVander(CurveReparametrized):
    @property
    def params(self):
        pred_y = [self.function(x) for x in self.x_points]
        return tuple(
            [y if isinstance(y, float) else y[np.argmin(np.abs(y - yp))] for y, yp in zip(pred_y, self.y_points)])


# vandermonde inverse matrix of np.linalg.inv(np.vander(np.linspace(-0.5, 0.5, num=3))[:, ::-1]), first c,
# second b third a of ax**2+bx+c
class CurveVandermondePolynomial(CurveVander, CurvePolynomial):
    def __init__(self, x_points, y_points, value_up=0, value_down=1, degree=1):
        self.degree = degree
        super().__init__(x_points, y_points, value_down=value_down, value_up=value_up)

    def new_params2natural_params(self, x_points, y_points):
        return np.linalg.lstsq(
            np.vander(x_points, N=self.degree + 1, increasing=True),
            y_points.reshape((-1, 1)),
            rcond=None)[0].ravel()
        # return np.ravel(get_inv_vandermonde_matrix(x_points) @ y_points.reshape((-1, 1)))


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


class CurveVanderCircle(CurveVander, CurveCircle):
    def new_params2natural_params(self, x_points, y_points):
        return points2circle(*np.array(list(zip(x_points, y_points))))
