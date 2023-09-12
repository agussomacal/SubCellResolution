from functools import partial
from typing import Dict, Generator, Tuple, Callable, Type

import numpy as np

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import ArrayIndexerNd
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.CellCreators.CellCreatorBase import CellBase
from lib.CellCreators.CurveCellCreators.CurveCellCreatorBase import CurveCellCreatorBase, map2unidimensional, \
    get_x_points, get_values_up_down, prepare_stencil4one_dimensionalization
from lib.CellCreators.CurveCellCreators.ParametersCurveCellCreators import DefaultCircleCurveCellCreator
from lib.Curves.AverageCurves import CurveAveragePolynomial
from lib.Curves.Curves import Curve, CurveReparametrized
from lib.Curves.CurveCircle import get_concavity, CurveSemiCircle, CircleParams
from lib.Curves.VanderCurves import CurveVandermondePolynomial, CurveVanderCircle
from lib.StencilCreators import Stencil


def smooth2_avg(v):
    return np.array([(vi + vip) / 2 for vi, vip in zip(v[:-1], v[1:])])


class ValuesCurveCellCreator(CurveCellCreatorBase):
    def __init__(self, vander_curve: Type[CurveReparametrized], regular_opposite_cell_searcher: Callable,
                 natural_params=False):
        super().__init__(regular_opposite_cell_searcher)
        self.vander_curve = vander_curve
        self.natural_params = natural_params

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[Curve, None, None]:
        value_up, value_down = get_values_up_down(coords, regular_opposite_cells)
        stencil_values = prepare_stencil4one_dimensionalization(independent_axis, value_up, value_down, stencil,
                                                                smoothness_index, indexer)
        stencil_values = stencil_values.sum(axis=1)
        x_points = get_x_points(stencil, independent_axis)
        curve = self.vander_curve(
            x_points=x_points,
            y_points=stencil_values,
            value_up=value_up,
            value_down=value_down,
            center=np.argmin(np.abs(x_points - coords[independent_axis] - 0.5))
        )
        curve.set_y_shift(np.min(stencil.coords[:, 1 - independent_axis]))
        if self.natural_params:
            yield curve.get_natural_parametrization_curve()
        else:
            yield curve


class ValuesLinearCellCreator(ValuesCurveCellCreator):
    def __init__(self, regular_opposite_cell_searcher: Callable, natural_params=False, avg_method=False):
        super().__init__(partial(CurveAveragePolynomial if avg_method else CurveVandermondePolynomial, degree=1),
                         regular_opposite_cell_searcher, natural_params)

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[Curve, None, None]:
        value_up = regular_opposite_cells[1].evaluate(coords.coords)
        value_down = regular_opposite_cells[0].evaluate(coords.coords)
        x_points, stencil_values = map2unidimensional(value_up, value_down, independent_axis, stencil)

        curve = self.vander_curve(
            x_points=smooth2_avg(x_points),
            y_points=smooth2_avg(stencil_values),
            value_up=value_up,
            value_down=value_down
        )
        if self.natural_params:
            yield curve.get_natural_parametrization_curve()
        else:
            yield curve


class ValuesCircleCellCreator(CurveCellCreatorBase):
    def __init__(self, regular_opposite_cell_searcher: Callable, natural_params=False):
        super().__init__(regular_opposite_cell_searcher)
        self.natural_params = natural_params

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[Curve, None, None]:
        value_up = regular_opposite_cells[1].evaluate(coords.coords)
        value_down = regular_opposite_cells[0].evaluate(coords.coords)
        x_points, stencil_values = map2unidimensional(value_up, value_down, independent_axis, stencil)
        concavity = get_concavity(x_points, stencil_values)
        curve = CurveVanderCircle(
            x_points=x_points,
            y_points=stencil_values,
            value_up=value_up,
            value_down=value_down,
        )
        if self.natural_params:
            yield curve.get_natural_parametrization_curve()
        else:
            yield curve


class ValuesDefaultCurveCellCreator(CurveCellCreatorBase):
    def __init__(self, vander_curve: Type[CurveReparametrized], regular_opposite_cell_searcher: Callable):
        super().__init__(regular_opposite_cell_searcher)
        self.vander_curve = vander_curve

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[Curve, None, None]:
        value_up = regular_opposite_cells[1].evaluate(coords.coords)
        value_down = regular_opposite_cells[0].evaluate(coords.coords)
        x_points = get_x_points(stencil, independent_axis)

        yield self.vander_curve(
            x_points=x_points,
            y_points=np.array([coords[1 - independent_axis] + 0.5] * len(x_points)),
            value_up=value_up,
            value_down=value_down,
        )


class ValuesDefaultLinearCellCreator(CurveCellCreatorBase):
    def __init__(self, regular_opposite_cell_searcher: Callable):
        super().__init__(regular_opposite_cell_searcher)

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[Curve, None, None]:
        value_up = regular_opposite_cells[1].evaluate(coords.coords)
        value_down = regular_opposite_cells[0].evaluate(coords.coords)
        x_points = get_x_points(stencil, independent_axis)

        yield CurveVandermondePolynomial(
            x_points=smooth2_avg(x_points),
            y_points=np.array([coords[1 - independent_axis] + 0.5] * 2),
            value_up=value_up,
            value_down=value_down,
            degree=1
        )


class ValuesDefaultCircleCellCreator(DefaultCircleCurveCellCreator):
    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[Curve, None, None]:
        circle = next(super(ValuesDefaultCircleCellCreator, self).create_curves(
            average_values=average_values,
            indexer=indexer, cells=cells,
            coords=coords,
            smoothness_index=smoothness_index,
            independent_axis=independent_axis,
            stencil=stencil,
            regular_opposite_cells=regular_opposite_cells))
        x_points = get_x_points(stencil, independent_axis)

        yield CurveVanderCircle(
            x_points=x_points,
            y_points=circle.function(x_points),
            value_up=circle.value_up,
            value_down=circle.value_down,
        )
