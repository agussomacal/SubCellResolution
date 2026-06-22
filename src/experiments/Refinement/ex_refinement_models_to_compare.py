import operator
from functools import partial

from experiments.global_params import CCExtraWeight, CurveAverageQuadraticCC
from lib.AuxiliaryStructures.Constants import REGULAR_CELL, CURVE_CELL
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_regular_cells_by_minmax
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesLineConsistentCurveCellCreator, \
    ValuesCurveCellCreator
from lib.CellCreators.RegularCellCreator import MirrorCellCreator, PiecewiseConstantRegularCellCreator
from lib.CellIterators import iterate_all, iterate_by_reconstruction_error_and_smoothness, \
    iterate_by_condition_on_smoothness
from lib.CellOrientators import BaseOrientator, OrientByGradient
from lib.SmoothnessCalculators import naive_piece_wise
from lib.StencilCreators import StencilCreatorFixedShape, StencilCreatorAdaptive
from lib.SubCellReconstruction import ReconstructionErrorMeasure, keep_cells_on_condition, curve_condition, \
    ReconstructionErrorMeasureDefaultStencil, CellCreatorPipeline, SubCellReconstruction

# ========== ========== Reconstruction error ========== ========== #
reconstruction_error_measure_default = ReconstructionErrorMeasure(
    StencilCreatorFixedShape((3, 3)),
    metric=2, central_cell_extra_weight=0)
reconstruction_error_measure_w = ReconstructionErrorMeasureDefaultStencil(
    StencilCreatorFixedShape((3, 3)),
    metric=2, central_cell_extra_weight=CCExtraWeight)

# ========== ========== Models definitions ========== ========== #
# piecewise01 = CellCreatorPipeline(
#     cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=REGULAR_CELL,
#                           condition=operator.eq),  # only regular cells
#     orientator=BaseOrientator(dimensionality=2),
#     stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
#     cell_creator=MirrorCellCreator(dimensionality=2)
# )
piecewise = CellCreatorPipeline(
    cell_iterator=iterate_all,  # only regular cells
    orientator=BaseOrientator(dimensionality=2),
    stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
    cell_creator=MirrorCellCreator(dimensionality=2)
)
piecewise01 = CellCreatorPipeline(
    cell_iterator=partial(iterate_by_condition_on_smoothness, value=REGULAR_CELL,
                          condition=operator.eq),  # only regular cells
    orientator=BaseOrientator(dimensionality=2),
    stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
    cell_creator=PiecewiseConstantRegularCellCreator(
        apriori_up_value=1, apriori_down_value=0, dimensionality=2)
)


def elvira_cc(angle_threshold,
              reconstruction_error_measure: ReconstructionErrorMeasure = reconstruction_error_measure_w):
    return CellCreatorPipeline(
        cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                              condition=operator.eq),
        orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2, method="sobel",
                                    angle_threshold=angle_threshold),
        stencil_creator=StencilCreatorFixedShape((3, 3)),
        cell_creator=ELVIRACurveCellCreator(
            regular_opposite_cell_searcher=get_opposite_regular_cells_by_minmax),
        reconstruction_error_measure=reconstruction_error_measure
    )


def aero_l_cc(angle_threshold,
              reconstruction_error_measure: ReconstructionErrorMeasure = reconstruction_error_measure_w):
    return CellCreatorPipeline(
        cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                              condition=operator.eq),
        orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2, method="sobel",
                                    angle_threshold=angle_threshold),
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                               independent_dim_stencil_size=3),
        cell_creator=ValuesLineConsistentCurveCellCreator(ccew=CCExtraWeight,
                                                          regular_opposite_cell_searcher=get_opposite_regular_cells_by_minmax),
        reconstruction_error_measure=reconstruction_error_measure
    )


def aero_l(angle_threshold, reconstruction_error_measure: ReconstructionErrorMeasure = reconstruction_error_measure_w):
    return CellCreatorPipeline(
        cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                              condition=operator.eq),
        orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2, method="sobel",
                                    angle_threshold=angle_threshold),
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                               independent_dim_stencil_size=3),
        cell_creator=ValuesLineConsistentCurveCellCreator(
            regular_opposite_cell_searcher=get_opposite_regular_cells_by_minmax),
        reconstruction_error_measure=reconstruction_error_measure
    )


def aero_q(angle_threshold, reconstruction_error_measure: ReconstructionErrorMeasure = reconstruction_error_measure_w):
    return CellCreatorPipeline(
        cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                              condition=operator.eq),
        orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2, method="sobel",
                                    angle_threshold=angle_threshold),
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                               independent_dim_stencil_size=3),
        cell_creator=ValuesCurveCellCreator(
            vander_curve=CurveAverageQuadraticCC,
            regular_opposite_cell_searcher=get_opposite_regular_cells_by_minmax),
        reconstruction_error_measure=reconstruction_error_measure
    )


def elvira(smoothness_calculator=naive_piece_wise, refinement=1, angle_threshold=0, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            elvira_cc(angle_threshold=angle_threshold,
                      reconstruction_error_measure=reconstruction_error_measure_default)
        ],
        obera_iterations=0
    )


def elvira_w(smoothness_calculator=naive_piece_wise, refinement=1, angle_threshold=0, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            elvira_cc(angle_threshold=angle_threshold,
                      reconstruction_error_measure=reconstruction_error_measure_w)
        ],
        obera_iterations=0
    )


def aero_linear_w(smoothness_calculator=naive_piece_wise, refinement=1, angle_threshold=0, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            aero_l_cc(angle_threshold=angle_threshold, reconstruction_error_measure=reconstruction_error_measure_w)
        ],
        obera_iterations=0
    )


def aero_linear(smoothness_calculator=naive_piece_wise, refinement=1, angle_threshold=0, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            aero_l(angle_threshold=angle_threshold, reconstruction_error_measure=reconstruction_error_measure_default)
        ],
        obera_iterations=0
    )


def quadratic(smoothness_calculator=naive_piece_wise, refinement=1, angle_threshold=0, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            aero_q(angle_threshold=angle_threshold)
        ],
        obera_iterations=0
    )
