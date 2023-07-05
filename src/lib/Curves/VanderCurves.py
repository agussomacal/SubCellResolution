import numpy as np

from lib.Curves.CurveBase import CurveReparametrized
from lib.Curves.CurveCircle import CurveSemiCircle, get_concavity
from lib.Curves.CurvePolynomial import CurvePolynomial


class CurveVander(CurveReparametrized):
    @property
    def params(self):
        pred_y = [self.function(x) for x in self.x_points]
        return tuple(
            [y if isinstance(y, float) else y[np.argmin(np.abs(y - yp))] for y, yp in zip(pred_y, self.y_points)])

    @params.setter
    def params(self, args):
        super(CurveVander, self.__class__).params.fset(self,
                                                       self.new_params2natural_params(self.x_points, args))


# vandermonde inverse matrix of np.linalg.inv(np.vander(np.linspace(-0.5, 0.5, num=3))[:, ::-1]), first c,
# second b third a of ax**2+bx+c
class CurveVandermondePolynomial(CurveVander, CurvePolynomial):
    def __init__(self, x_points, y_points, value_up=0, value_down=1, degree=1, ccew=0, center=None):
        self.degree = degree
        super().__init__(x_points, y_points, value_down=value_down, value_up=value_up, ccew=ccew, center=center)

    def new_params2natural_params(self, x_points, y_points):
        return np.linalg.lstsq(
            np.vander(x_points, N=self.degree + 1, increasing=True),
            y_points.reshape((-1, 1)),
            rcond=None)[0].ravel()
        # return np.ravel(get_inv_vandermonde_matrix(x_points) @ y_points.reshape((-1, 1)))

    def get_natural_parametrization_curve(self):
        return CurvePolynomial(self.new_params2natural_params(self.x_points, self.y_points), value_up=self.value_up,
                               value_down=self.value_down, x_shift=self.x_shift)


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


class CurveVanderCircle(CurveVander, CurveSemiCircle):
    def __init__(self, x_points, y_points, value_up=0, value_down=1, concave=False, ccew=0, center=None):
        super().__init__(x_points=x_points, y_points=y_points, value_up=value_up, value_down=value_down, ccew=ccew,
                         center=center)
        self.concave = concave

    def new_params2natural_params(self, x_points, y_points):
        self.concave = get_concavity(x_points, y_points) > 0
        return points2circle(*np.array(list(zip(x_points, y_points))))

    def get_natural_parametrization_curve(self):
        return CurveSemiCircle(self.new_params2natural_params(self.x_points, self.y_points), value_up=self.value_up,
                               value_down=self.value_down, concave=self.concave)
