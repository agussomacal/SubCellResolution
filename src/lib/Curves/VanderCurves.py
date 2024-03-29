import numpy as np

from lib.Curves.CurveCircle import CurveSemiCircle, get_concavity
from lib.Curves.CurvePolynomial import CurvePolynomial
from lib.Curves.Curves import CurveReparametrized


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
    def __init__(self, x_points, y_points, value_up=0, value_down=1, degree=1, ccew=0, center=None, weights=None):
        self.degree = degree
        super().__init__(x_points, y_points, value_down=value_down, value_up=value_up, ccew=ccew, center=center,
                         weights=weights)

    def new_params2natural_params(self, x_points, y_points):
        return np.linalg.lstsq(
            np.vander(x_points, N=self.degree + 1, increasing=True) * self.weights[:, np.newaxis],
            (y_points * self.weights).reshape((-1, 1)),
            rcond=None)[0].ravel()

    def get_natural_parametrization_curve(self):
        return CurvePolynomial(self.new_params2natural_params(self.x_points, self.y_points), value_up=self.value_up,
                               value_down=self.value_down)


def points2circle(a, b, c):
    a, b, c = list(map(np.array, [a, b, c]))
    matrix = np.array([b - a,
                       c - a,
                       c - b])
    try:
        x0, y0 = np.linalg.lstsq(matrix,
                                 [(a + b) / 2 @ (b - a),
                                  (a + c) / 2 @ (c - a),
                                  (c + b) / 2 @ (c - b)],
                                 rcond=None)[0]
        return x0, y0, \
            (np.sqrt((a[0] - x0) ** 2 + (a[1] - y0) ** 2) +
             np.sqrt((b[0] - x0) ** 2 + (b[1] - y0) ** 2) +
             np.sqrt((c[0] - x0) ** 2 + (c[1] - y0) ** 2)) / 3
    except:
        # if it can't solve there is a problem in the transfomration
        x0, y0 = tuple(np.nanmean([a, b, c], axis=0).tolist())
        return x0, y0, 1


class CurveVanderCircle(CurveVander, CurveSemiCircle):
    def __init__(self, x_points, y_points, value_up=0, value_down=1, ccew=0, center=None, weights=None):
        super().__init__(x_points=x_points, y_points=y_points, value_up=value_up, value_down=value_down, ccew=ccew,
                         center=center, weights=weights)
        self.concave = get_concavity(x_points, y_points) > 0  # concave

    def new_params2natural_params(self, x_points, y_points):
        self.concave = get_concavity(x_points, y_points) > 0
        return points2circle(*np.array(list(zip(x_points, y_points))))

    def get_natural_parametrization_curve(self):
        return CurveSemiCircle(self.new_params2natural_params(self.x_points, self.y_points), value_up=self.value_up,
                               value_down=self.value_down, concave=self.concave)
