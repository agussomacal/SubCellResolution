import numpy as np
import scipy
from scipy.optimize import minimize, LinearConstraint

from lib.Curves.Curves import CurveReparametrized
from lib.Curves.CurvePolynomial import CurvePolynomial


class CurveAvg(CurveReparametrized):
    @property
    def params(self):
        """
        :return: The average of the 1d-cells
        """
        return tuple([self.function_integral(x + 0.5) - self.function_integral(x - 0.5) for x in self.x_points])


def constrained_weighted_ls(V, weights, y_points, constraints):
    """
    constraints: dict with keys 'A_c' (constraint matrix) and 'b_c' (constraint RHS)
                 representing A_c @ x = b_c
    """

    # Initial guess (unconstrained solution)
    x_init = np.linalg.lstsq(V * weights[:, np.newaxis], (y_points * weights).reshape(-1, 1), rcond=None)[0].ravel()
    # x_init = np.linalg.lstsq(V, y_points.reshape(-1, 1), rcond=None)[0].ravel()

    # Set up constraints
    # constr_dict = {'type': 'eq', 'fun': lambda x, A_c, b_c: A_c @ x - b_c,
    #                'args': (constraints['A_c'], constraints['b_c'])}
    #
    # Define the objective: sum of squared residuals (weighted)
    # def objective(x):
    #     residual = V @ x - y_points
    #     return np.sum(residual ** 2)
    #
    # result = minimize(objective, x_init, method='SLSQP', constraints=[constr_dict])
    # print(np.abs((constraints['A_c'] @ x_init - constraints['b_c']) / constraints['b_c']) >= np.abs(
    #     (constraints['A_c'] @ result.x - constraints['b_c']) / constraints['b_c']),
    #       np.linalg.norm(V @ x_init - V @ result.x) / np.linalg.norm(V @ x_init))
    # if np.abs((constraints['A_c'] @ x_init - constraints['b_c']) / constraints['b_c']) >= np.abs(
    #         (constraints['A_c'] @ result.x - constraints['b_c']) / constraints['b_c']):
    #     return result.x

    # Shift constant to satisfy area
    # x_init[0] += constraints['b_c'] - constraints['A_c'] @ x_init  # center to satisfy central area
    return x_init


class CurveAveragePolynomial(CurveReparametrized, CurvePolynomial):
    def __init__(self, x_points, y_points, value_up=0, value_down=1, degree=1, ccew=0, center=None, weights=None):
        """

        :param x_points:
        :param y_points:
        :param value_up:
        :param value_down:
        :param degree:
        :param ccew: central cell extra weight
        """
        self.degree = degree
        super().__init__(x_points, y_points, value_down=value_down, value_up=value_up, ccew=ccew, center=center,
                         weights=weights)

    def new_params2natural_params(self, x_points, y_points):
        V = (np.vander(x_points + 0.5, N=self.degree + 2, increasing=True)[:, 1:] -
             np.vander(x_points - 0.5, N=self.degree + 2, increasing=True)[:, 1:]) / np.arange(1, self.degree + 2)[
                np.newaxis, :]
        # return np.linalg.lstsq(V * self.weights[:, np.newaxis],
        #                        (y_points * self.weights).reshape((-1, 1)),
        #                        rcond=None)[0].ravel()
        # return np.linalg.pinv(V.T @ (V * self.weights[:, np.newaxis])) @ (V.T @ (y_points * self.weights))
        return constrained_weighted_ls(V, self.weights, y_points,
                                       {'A_c': V[self.center, :], 'b_c': y_points[self.center]})

    def get_natural_parametrization_curve(self):
        return CurvePolynomial(self.new_params2natural_params(self.x_points, self.y_points), value_up=self.value_up,
                               value_down=self.value_down)


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
