import operator
from functools import partial

from experiments.subcell_paper.global_params import CCExtraWeight, CurveAverageQuadraticCC, cgray, \
    cblue, cgreen, cred, corange
from lib.AuxiliaryStructures.Constants import REGULAR_CELL, CURVE_CELL
from lib.CellCreators.CurveCellCreators.CurveCellCreatorBase import get_values_up_down_01_harcoded
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_regular_cells_by_minmax
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesLineConsistentCurveCellCreator, \
    ValuesCurveCellCreator
from lib.CellCreators.CurveCellCreators.VertexCellCreator import LinearVertexCellCurveCellCreator
from lib.CellCreators.LearningFluxRegularCellCreator import LearningFluxRegularCellCreator
from lib.CellCreators.RegularCellCreator import MirrorCellCreator, PiecewiseConstantRegularCellCreator
from lib.CellCreators.VertexCellCreators.VertexCellCreatorBase import VertexCellCreatorUsingNeighboursLines
from lib.CellIterators import iterate_all, iterate_by_reconstruction_error_and_smoothness, \
    iterate_by_condition_on_smoothness
from lib.CellOrientators import BaseOrientator, OrientByGradient, OrientPredefined
from lib.SmoothnessCalculators import naive_piece_wise
from lib.StencilCreators import StencilCreatorFixedShape, StencilCreatorAdaptive
from lib.SubCellReconstruction import ReconstructionErrorMeasure, keep_cells_on_condition, curve_condition, \
    ReconstructionErrorMeasureDefaultStencil, CellCreatorPipeline, SubCellFlux, SubCellReconstruction

# ========== ========== Reconstruction error ========== ========== #
reconstruction_error_measure_default = ReconstructionErrorMeasure(
    StencilCreatorFixedShape((3, 3)),
    metric=2, central_cell_extra_weight=1)
reconstruction_error_measure_default_singular = ReconstructionErrorMeasure(
    StencilCreatorFixedShape((3, 3)),
    metric=2, central_cell_extra_weight=1,
    keeping_cells_condition=partial(keep_cells_on_condition, condition=curve_condition))


def reconstruction_error_measure_3x3_w(ccw=CCExtraWeight):
    return ReconstructionErrorMeasureDefaultStencil(
        StencilCreatorFixedShape((3, 3)),
        metric=2, central_cell_extra_weight=ccw)


reconstruction_error_measure_3x3_w_singular = ReconstructionErrorMeasureDefaultStencil(
    StencilCreatorFixedShape((3, 3)),
    metric=2, central_cell_extra_weight=CCExtraWeight,
    keeping_cells_condition=partial(keep_cells_on_condition, condition=curve_condition))

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


def elvira_cc(angle_threshold, weight=CCExtraWeight):
    return CellCreatorPipeline(
        cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                              condition=operator.eq),
        orientator=OrientByGradient(kernel_size=(5, 5), dimensionality=2, method="optim",
                                    angle_threshold=angle_threshold),
        stencil_creator=StencilCreatorFixedShape((3, 3)),
        cell_creator=ELVIRACurveCellCreator(
            regular_opposite_cell_searcher=get_opposite_regular_cells_by_minmax),
        reconstruction_error_measure=reconstruction_error_measure_3x3_w(weight)
    )


def aero_l(angle_threshold):
    return CellCreatorPipeline(
        cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                              condition=operator.eq),
        # orientator=OrientPredefined(predefined_axis=0, dimensionality=2),
        orientator=OrientByGradient(kernel_size=(5, 5), dimensionality=2, method="optim",
                                    angle_threshold=angle_threshold),
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                               independent_dim_stencil_size=3),
        cell_creator=ValuesLineConsistentCurveCellCreator(ccew=CCExtraWeight,
                                                          regular_opposite_cell_searcher=get_opposite_regular_cells_by_minmax),
        reconstruction_error_measure=reconstruction_error_measure_3x3_w()
    )


def aero_q(angle_threshold):
    return CellCreatorPipeline(
        cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                              condition=operator.eq),
        orientator=OrientByGradient(kernel_size=(5, 5), dimensionality=2, method="optim",
                                    angle_threshold=angle_threshold),
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                               independent_dim_stencil_size=3),
        cell_creator=ValuesCurveCellCreator(
            vander_curve=CurveAverageQuadraticCC,
            regular_opposite_cell_searcher=get_opposite_regular_cells_by_minmax),
        reconstruction_error_measure=reconstruction_error_measure_3x3_w()
    )


tem = CellCreatorPipeline(
    cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                          condition=operator.eq),
    orientator=OrientPredefined(predefined_axis=0),
    stencil_creator=StencilCreatorFixedShape(stencil_shape=(3, 3)),
    cell_creator=VertexCellCreatorUsingNeighboursLines(
        regular_opposite_cell_searcher=get_opposite_regular_cells_by_minmax,
    ),
    reconstruction_error_measure=reconstruction_error_measure_3x3_w()
)


def nn(learning_manager):
    return CellCreatorPipeline(
        cell_iterator=partial(iterate_by_condition_on_smoothness, value=CURVE_CELL, condition=operator.eq),
        orientator=OrientByGradient(kernel_size=(5, 5), dimensionality=2, method="optim",
                                    angle_threshold=45),
        stencil_creator=StencilCreatorFixedShape(stencil_shape=(3, 3)),
        cell_creator=LearningFluxRegularCellCreator(learning_manager=learning_manager,
                                                    updown_value_getter=get_values_up_down_01_harcoded),
        reconstruction_error_measure=reconstruction_error_measure_3x3_w()
    )


# ========== ========== Models definitions ========== ========== #
def nn_flux(smoothness_calculator=naive_piece_wise, learning_manager=None, *args, **kwargs):
    return SubCellFlux(
        name="All",
        smoothness_calculator=smoothness_calculator,
        cell_creators=
        [
            piecewise01,
            nn(learning_manager)
        ],
    )


def upwind(smoothness_calculator=naive_piece_wise, refinement=1, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise
        ],
        obera_iterations=0
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
            elvira_cc(angle_threshold=angle_threshold)
        ],
        obera_iterations=0
    )


def elvira_oriented(smoothness_calculator=naive_piece_wise, refinement=1, angle_threshold=45, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            elvira_cc(angle_threshold=angle_threshold)
        ],
        obera_iterations=0
    )


def aero_linear_oriented(smoothness_calculator=naive_piece_wise, refinement=1, angle_threshold=45, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            aero_l(angle_threshold=angle_threshold)
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
            aero_l(angle_threshold=angle_threshold)
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


def qelvira(smoothness_calculator=naive_piece_wise, refinement=1, angle_threshold=0):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            elvira_cc(angle_threshold=angle_threshold),
            aero_q(angle_threshold=angle_threshold)
        ],
        obera_iterations=0
    )


def aero_lq(smoothness_calculator=naive_piece_wise, refinement=1, angle_threshold=0):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            aero_l(angle_threshold=angle_threshold),
            aero_q(angle_threshold=angle_threshold)
        ],
        obera_iterations=0
    )


def aero_qelvira_vertex(smoothness_calculator=naive_piece_wise, refinement=1, angle_threshold=0, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            elvira_cc(angle_threshold),
            tem,
            aero_q(angle_threshold=angle_threshold),
            # ------------ AVRO ------------ #
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientByGradient(kernel_size=(5, 5), dimensionality=2, method="optim",
                                            angle_threshold=angle_threshold),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=4),
                cell_creator=LinearVertexCellCurveCellCreator(
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_minmax),
                reconstruction_error_measure=reconstruction_error_measure_3x3_w()
            )
        ],
        obera_iterations=0
    )


# def ml_vql(smoothness_calculator=naive_piece_wise, refinement=1, angle_threshold=0, *args, **kwargs):
#     return SubCellReconstruction(
#         name="All",
#         smoothness_calculator=smoothness_calculator,
#         cell_classifier=partial(cell_classifier_ml, ml_model=curve_classification_ml_model,
#                                 regular_cell_creators_indexes=[0]),
#         reconstruction_error_measure=reconstruction_error_measure_default,
#         refinement=refinement,
#         cell_creators=
#         [
#             piecewise01,
#             # aero_l(angle_threshold),
#             elvira_cc(angle_threshold=angle_threshold),
#             aero_q(angle_threshold=angle_threshold),
#             tem,
#         ],
#         obera_iterations=0
#     )


winner_color_dict = {
    'PolynomialCelldegree (1, 1)': cgray,
    'CellCurveBaseCurveAveragePolynomialLine': cblue,
    "CellCurveBaseCurvePolynomialLine": cblue,
    'CellCurveBaseCurveAverageQuadraticCCQuadratic': cgreen,
    'CellCurveBaseVertexLinearExtendedVertexCurvePolynomialByPartsLineaddCurvePolynomialByPartsLine': cred,
    'CellCurveBaseCurveVertexPolynomialVertexCurvePolynomialByPartsLineaddCurvePolynomialByPartsLine': corange,
    'CellCurveBaseVertexLinearExtendedVertexCurvePolynomialByPartsLineaddCurvePolynomialByPartsLineaddNoCurveRegionNoCurveRegion': corange
}
