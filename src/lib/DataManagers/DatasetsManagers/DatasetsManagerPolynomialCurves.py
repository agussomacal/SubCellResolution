from pathlib import Path
from typing import Union, Tuple

import numpy as np
from numpy.polynomial import Polynomial
from scipy.linalg import lstsq
from sklearn.preprocessing import PolynomialFeatures

from lib.Curves.CurveBase import CurveBase
from lib.Curves.CurvePolynomial import CurvePolynomial
from lib.SubCellSchemes.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.SubCellSchemes.DataManagers.DatasetsManagers.DatasetsBaseManager import DatasetsBaseManager


class DatasetsManagerPolynomialCurves(DatasetsBaseManager):
    def __init__(self, path2data: Union[str, Path], N: int, kernel_size: Tuple[int, int], min_val: float,
                 max_val: float, workers=np.Inf, recalculate=False, output_degree: int = 1,
                 velocity_range: Tuple[Tuple, Tuple] = ((1e-10, 0), (1.0, 0)), curve_position_radius: float = 1):
        self.output_degree = output_degree
        self.curve_position_radius = curve_position_radius
        super().__init__(path2data=path2data, N=N, kernel_size=kernel_size, min_val=min_val, max_val=max_val,
                         recalculate=recalculate, workers=workers, curve=CurvePolynomial,
                         velocity_range=velocity_range)

    @property
    def base_name(self):
        return "{}_deg{}".format(super(DatasetsManagerPolynomialCurves, self).base_name, self.output_degree)

    def get_curve_data(self):
        x_points = np.append(self.center[0], np.random.uniform(self.kernel_size[0], size=self.output_degree))
        y_points = np.append(np.random.uniform(
            self.center[1] - self.curve_position_radius,
            self.center[1] + self.curve_position_radius),
            np.random.uniform(self.kernel_size[1], size=self.output_degree)
        )
        x_transformed = PolynomialFeatures(self.output_degree).fit_transform(x_points.reshape((-1, 1)))
        poly_coefs = lstsq(x_transformed, y_points)[0]

        return poly_coefs

    # --------- predict/find curve ---------- #
    def create_curve_from_predictions(self, model_predictions, coords: CellCoords, independent_axis: int) -> CurveBase:
        model_predictions[0] += coords.coords[1 - independent_axis] + 0.5 - self.center[1]
        return CurvePolynomial(Polynomial(model_predictions),
                               x_shift=coords.coords[independent_axis] + 0.5 - self.center[0])
