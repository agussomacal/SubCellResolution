from functools import partial
from typing import Dict, Generator, Tuple, Callable, Type

import numpy as np

from PerplexityLab.miscellaneous import ClassPartialInit
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import ArrayIndexerNd
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.CellCreators.CellCreatorBase import CellBase
from lib.CellCreators.CurveCellCreators.CurveCellCreatorBase import CurveCellCreatorBase, get_x_points, \
    prepare_stencil4one_dimensionalization, get_values_up_down_regcell_eval
from lib.CellCreators.CurveCellCreators.ParametersCurveCellCreators import DefaultCircleCurveCellCreator
from lib.Curves.AverageCurves import CurveAveragePolynomial
from lib.Curves.Curves import Curve, CurveReparametrized
from lib.Curves.VanderCurves import CurveVandermondePolynomial, CurveVanderCircle
from lib.StencilCreators import Stencil


def smooth2_avg(v):
    return np.array([(vi + vip) / 2 for vi, vip in zip(v[:-1], v[1:])])


class ValuesCurveCellCreator(CurveCellCreatorBase):
    def __init__(self, vander_curve: Type[CurveReparametrized], regular_opposite_cell_searcher: Callable,
                 natural_params=False, updown_value_getter: Callable = get_values_up_down_regcell_eval):
        super().__init__(regular_opposite_cell_searcher, updown_value_getter=updown_value_getter)
        self.vander_curve = vander_curve
        self.natural_params = natural_params

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[Curve, None, None]:
        value_up, value_down = self.updown_value_getter(coords, regular_opposite_cells)
        stencil_values = prepare_stencil4one_dimensionalization(independent_axis, value_up, value_down, stencil,
                                                                smoothness_index, indexer)
        stencil_values = stencil_values.sum(axis=1)
        x_points = get_x_points(stencil, independent_axis)
        curve = self.vander_curve(
            x_points=x_points,
            y_points=stencil_values,
            value_up=value_up,
            value_down=value_down,
            center=np.argmin(np.abs(x_points - coords[independent_axis] - 0.5)),
            weights=np.exp(-(x_points - coords[independent_axis] - 0.5) ** 2)
            # weights=1 * ((np.abs(x_points - coords[independent_axis] - 0.5)) < (len(x_points) // 2 + 1))
        )
        # TODO: Solve this better, some problem in certain stencils create polynomial deg 2 with nan coefs
        if not np.any(np.isnan(curve.params)):
            curve.set_y_shift(np.min(stencil.coords[:, 1 - independent_axis]))
            if self.natural_params:
                yield curve.get_natural_parametrization_curve()
            else:
                yield curve


class ValuesLineConsistentCurveCellCreator(ValuesCurveCellCreator):
    def __init__(self, regular_opposite_cell_searcher: Callable,
                 natural_params=False, ccew=0, updown_value_getter: Callable = get_values_up_down_regcell_eval):
        super().__init__(vander_curve=ClassPartialInit(CurveAveragePolynomial, class_name="CurveAverageLinearCC",
                                                       degree=1, ccew=ccew),
                         regular_opposite_cell_searcher=regular_opposite_cell_searcher,
                         natural_params=natural_params,
                         updown_value_getter=updown_value_getter)

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[Curve, None, None]:
        value_up, value_down = self.updown_value_getter(coords, regular_opposite_cells)
        stencil_values = prepare_stencil4one_dimensionalization(independent_axis, value_up, value_down, stencil,
                                                                smoothness_index, indexer)
        stencil_values = stencil_values.sum(axis=1)
        x_points = get_x_points(stencil, independent_axis)
        curve = self.vander_curve(
            x_points=x_points,
            y_points=stencil_values,
            value_up=value_up,
            value_down=value_down,
            center=np.argmin(np.abs(x_points - coords[independent_axis] - 0.5)),
            # weights=1/(1+np.abs(x_points - coords[independent_axis] - 0.5))**2
            weights=1 * ((np.abs(x_points - coords[independent_axis] - 0.5)) < 2)
        )
        curve.set_y_shift(stencil_values[curve.center] - curve.function(curve.x_points[curve.center]))
        curve.set_y_shift(np.min(stencil.coords[:, 1 - independent_axis]))
        if self.natural_params:
            yield curve.get_natural_parametrization_curve()
        else:
            yield curve


class ValuesLinearCellCreator(ValuesCurveCellCreator):
    def __init__(self, regular_opposite_cell_searcher: Callable, natural_params=False, avg_method=False,
                 updown_value_getter: Callable = get_values_up_down_regcell_eval):
        super().__init__(ClassPartialInit(CurveAveragePolynomial if avg_method else CurveVandermondePolynomial,
                                          class_name="CurveAverageLinear" if avg_method else "CurveVanderLinear",
                                          degree=1),
                         regular_opposite_cell_searcher, natural_params, updown_value_getter=updown_value_getter)

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[Curve, None, None]:
        value_up, value_down = self.updown_value_getter(coords, regular_opposite_cells)
        stencil_values = prepare_stencil4one_dimensionalization(value_up, value_down, independent_axis, stencil,
                                                                smoothness_index, indexer)
        stencil_values = stencil_values.sum(axis=1)
        x_points = get_x_points(stencil, independent_axis)
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
    def __init__(self, regular_opposite_cell_searcher: Callable, natural_params=False,
                 updown_value_getter: Callable = get_values_up_down_regcell_eval):
        super().__init__(regular_opposite_cell_searcher, updown_value_getter=updown_value_getter)
        self.natural_params = natural_params

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[Curve, None, None]:
        value_up, value_down = self.updown_value_getter(coords, regular_opposite_cells)
        stencil_values = prepare_stencil4one_dimensionalization(value_up, value_down, independent_axis, stencil,
                                                                smoothness_index, indexer)
        stencil_values = stencil_values.sum(axis=1)
        x_points = get_x_points(stencil, independent_axis)
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
    def __init__(self, vander_curve: Type[CurveReparametrized], regular_opposite_cell_searcher: Callable,
                 updown_value_getter: Callable = get_values_up_down_regcell_eval):
        super().__init__(regular_opposite_cell_searcher, updown_value_getter=updown_value_getter)
        self.vander_curve = vander_curve

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[Curve, None, None]:
        value_up, value_down = self.updown_value_getter(coords, regular_opposite_cells)
        x_points = get_x_points(stencil, independent_axis)
        yield self.vander_curve(
            x_points=x_points,
            y_points=np.array([coords[1 - independent_axis] + 0.5] * len(x_points)),
            value_up=value_up,
            value_down=value_down,
        )


class ValuesDefaultLinearCellCreator(CurveCellCreatorBase):
    def __init__(self, regular_opposite_cell_searcher: Callable,
                 updown_value_getter: Callable = get_values_up_down_regcell_eval):
        super().__init__(regular_opposite_cell_searcher, updown_value_getter=updown_value_getter)

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[Curve, None, None]:
        value_up, value_down = self.updown_value_getter(coords, regular_opposite_cells)
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
