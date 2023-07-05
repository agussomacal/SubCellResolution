import operator
from functools import partial
from typing import Tuple, List, Callable, Union

import numpy as np

NUMERICAL_DERIVATIVE_STEP = 1e-5


def calculate_integration_breakpoints_in_rectangle(x_limits: Tuple[float], y_limits: Tuple[float],
                                                   inverse: Callable[[float], List[float]]):
    x_breakpoints = list(x_limits)
    for y in y_limits:
        for x in inverse(y):  # because the inverse may not be unique
            if x_limits[0] <= x <= x_limits[1]:
                x_breakpoints.append(x)
    return sorted(list(set(x_breakpoints)))


def new_calculate_integrals(x_limits: Tuple[float], y_limits: Tuple[float], self, other, operator_function) -> float:
    return operator_function(self.calculate_integrals(x_limits, y_limits),
                             other.calculate_integrals(x_limits, y_limits))


def new_function_inverse(y: float, self, other) -> float:
    return np.sort(np.unique(np.append(self.function_inverse(y), other.function_inverse(y)))).tolist()


def new_function(x: Union[float, np.ndarray], self, other):
    xsize = tuple(np.append(np.squeeze(np.shape(np.array([x]))), -1))
    return np.concatenate((np.reshape(self.function(x), xsize),
                           np.reshape(other.function(x), xsize)),
                          axis=1)
    # return np.array(
    #     [np.append(np.ravel([f1]), np.ravel([f2])) for f1, f2 in
    #      zip(np.reshape(self.function(x), (len(x), -1)), np.reshape(other.function(x), (len(x), -1)))])


def new_evaluate(x: Union[float, np.ndarray], y: Union[float, np.ndarray] = None, self=None, other=None,
                 operator_function=None):
    return operator_function(self.evaluate(x, y), other.evaluate(x, y))


def create_new_curve(self, other, operator_function: Callable[[float, float], float]):
    if isinstance(other, CurveBase):
        new_curve = CurveBase(curve_name=f"{self}{operator_function.__name__}{other}")
        # the partial is needed to make the function picklable so it can be used in experiments and paralelize
        setattr(new_curve, "calculate_integrals",
                partial(new_calculate_integrals, self=self, other=other, operator_function=operator_function))
        setattr(new_curve, "function_inverse", partial(new_function_inverse, self=self, other=other))
        setattr(new_curve, "function", partial(new_function, self=self, other=other))
        setattr(new_curve, "evaluate",
                partial(new_evaluate, self=self, other=other, operator_function=operator_function))

        return new_curve
    else:
        raise Exception("Not implemented.")


class CurveBase:
    def __init__(self, curve_name="", value_up=0, value_down=1):
        self.curve_name = curve_name
        self.value_up = value_up
        self.value_down = value_down

    @property
    def params(self):
        raise Exception("Not implemented.")

    @params.setter
    def params(self, *args):
        raise Exception("Not implemented.")

    @property
    def dim(self):
        return len(self.params)

    def __call__(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray] = None):
        """
        Evaluate the Curve.
        """
        if isinstance(x, (list, np.ndarray)):
            reshape = False
        else:
            x = np.array([x])
            reshape = True

        if y is None:
            # it means we ask for the y value associated with the x queried. Then evaluate function.
            res = self.function(x)
        else:
            # it means we ask for the region of the points (x, y):
            # the question is if it is greater or lower than the curve.
            res = self.evaluate(x, y)

        return float(np.ravel(res)) if reshape else res

    def evaluate(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray] = None):
        # it means we ask for the region of the points (x, y):
        # the question is if it is greater or lower than the curve.
        fx = self.function(x)
        res = np.zeros(np.shape(fx))  # when the curve is not defined in the point, the value is 0 by default
        res[(~np.isnan(fx)) & (y >= fx)] = self.value_up
        # res[(~np.isnan(fx)) & (y == fx)] = (self.value_up + self.value_down) / 2
        res[(~np.isnan(fx)) & (y < fx)] = self.value_down
        return res

    def function(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Given x tells which is the value of y of the curve.
        """
        raise Exception("Not implemented.")

    def function_inverse(self, y: float) -> List[float]:
        raise Exception("Not implemented.")

    def function_integral(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise Exception("Not implemented.")

    def derivative(self, x: float):
        return (self.function(x + NUMERICAL_DERIVATIVE_STEP) - self.function(
            x - NUMERICAL_DERIVATIVE_STEP)) / 2 / NUMERICAL_DERIVATIVE_STEP

    def calculate_integrals(self, x_breakpoints, y_limits: Tuple[float, ...]) -> np.ndarray:
        dy = y_limits[1] - y_limits[0]
        x0 = np.reshape(x_breakpoints[:-1], (-1, 1))
        xf = np.reshape(x_breakpoints[1:], (-1, 1))
        dx = (xf - x0)
        integrals = self.function_integral(xf) - self.function_integral(x0) - y_limits[0] * dx
        integrals = np.min((integrals, dy * dx), axis=0)  # upper bound
        integrals *= (integrals > 0)  # lower bound (relu)
        integrals = self.value_down * integrals + (dy * dx - integrals) * self.value_up
        integrals[np.isnan(integrals)] = 0  # when the curve is not defined in the region, the area is 0
        return integrals

    def calculate_rectangle_average(self, x_limits: Tuple[float, ...], y_limits: Tuple[float, ...]) -> float:
        rectangle_area = np.product((np.diff(x_limits), np.diff(y_limits)))
        if rectangle_area == 0:
            return 0
        else:
            x_breakpoints = calculate_integration_breakpoints_in_rectangle(x_limits, y_limits, self.function_inverse)
            integrals = self.calculate_integrals(x_breakpoints, y_limits)
            return float(np.sum(integrals))

    def __str__(self):
        return self.__class__.__name__ + self.curve_name

    def __add__(self, other):
        return create_new_curve(self, other, operator.add)

    def __sub__(self, other):
        return create_new_curve(self, other, operator.sub)


class CurveReparametrized(CurveBase):
    """
    Abstract class with common methods for both average and point curves.
    """

    def __init__(self, x_points, y_points, value_up=0, value_down=1, ccew=0, center=None):
        """

        :param points: Nx2 matrix of points
        :param value_up:
        :param value_down:
        :param x_shift:
        """
        self.x_points = x_points
        self.y_points = y_points
        self.center = len(self.x_points) // 2 if center is None else center
        self.ccew = ccew
        self.weights = np.ones(len(x_points))
        self.weights[center] += self.ccew
        self.weights = np.sqrt(self.weights)
        super().__init__(self.new_params2natural_params(x_points, y_points), value_up, value_down)

    def new_params2natural_params(self, x_points, y_points):
        """
        How to pass from the new parametrization to the natural parametrization
        :param x_points:
        :param y_points:
        :return:
        """
        raise Exception("Not implemented.")

    def get_natural_parametrization_curve(self):
        raise Exception("Not implemented.")

    # @property
    # def params(self):
    #     raise Exception("Not implemented.")

    # @params.setter
    # def params(self, args):
    #     super(CurveReparametrized, self.__class__).params.fset(self,
    #                                                            self.new_params2natural_params(self.x_points, args))
