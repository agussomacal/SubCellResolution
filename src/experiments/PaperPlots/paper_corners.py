import operator
from functools import partial

import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial
from experiments.PaperPlots.paper_smooth_domains import fit_model
from experiments.subcell_paper.ex_aero import piecewise_constant, \
    quadratic_aero, quartic_aero, elvira, elvira_w_oriented, linear_obera, linear_obera_w, \
    quadratic_obera_non_adaptive
from experiments.subcell_paper.global_params import CurveAverageQuadraticCC, CCExtraWeight, cgray, cblue, cgreen, cred, \
    corange, cpurple, runsinfo
from experiments.subcell_paper.tools4binary_images import plot_reconstruction
from lib.AuxiliaryStructures.Constants import REGULAR_CELL, CURVE_CELL
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_regular_cells_by_minmax
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesCurveCellCreator
from lib.CellCreators.CurveCellCreators.VertexCellCreator import LinearVertexCellCurveCellCreator
from lib.CellCreators.RegularCellCreator import PiecewiseConstantRegularCellCreator
from lib.CellCreators.VertexCellCreators.VertexCellCreatorBase import VertexCellCreatorUsingNeighboursLines
from lib.CellIterators import iterate_by_reconstruction_error_and_smoothness, \
    iterate_by_condition_on_smoothness
from lib.CellOrientators import BaseOrientator, OrientByGradient, OrientPredefined
from lib.SmoothnessCalculators import naive_piece_wise
from lib.StencilCreators import StencilCreatorFixedShape, StencilCreatorAdaptive
from lib.SubCellReconstruction import ReconstructionErrorMeasure, ReconstructionErrorMeasureDefaultStencil, \
    CellCreatorPipeline, SubCellReconstruction

# ========== ========== Reconstruction error ========== ========== #
reconstruction_error_measure_default = ReconstructionErrorMeasure(
    StencilCreatorFixedShape((3, 3)),
    metric=2, central_cell_extra_weight=1)
reconstruction_error_measure_w = ReconstructionErrorMeasureDefaultStencil(
    StencilCreatorFixedShape((3, 3)),
    metric=2, central_cell_extra_weight=CCExtraWeight)

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


tem = CellCreatorPipeline(
    cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                          condition=operator.eq),
    orientator=OrientPredefined(predefined_axis=0),
    stencil_creator=StencilCreatorFixedShape(stencil_shape=(3, 3)),
    cell_creator=VertexCellCreatorUsingNeighboursLines(
        regular_opposite_cell_searcher=get_opposite_regular_cells_by_minmax,
    ),
    reconstruction_error_measure=reconstruction_error_measure_w
)


def aero_qelvira_vertex(smoothness_calculator=naive_piece_wise, refinement=1, obera_iterations=0, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            elvira_cc(angle_threshold=45),
            aero_q(angle_threshold=45),
            tem,
            # ------------ AVRO ------------ #
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2, method="sobel",
                                            angle_threshold=0),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=4),
                cell_creator=LinearVertexCellCurveCellCreator(
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_minmax),
                reconstruction_error_measure=reconstruction_error_measure_w
            ),
        ],
        obera_iterations=obera_iterations
    )


def aero_qvertex(smoothness_calculator=naive_piece_wise, refinement=1, obera_iterations=0, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            aero_q(angle_threshold=45),
            tem,
            # ------------ AVRO ------------ #
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2, method="sobel",
                                            angle_threshold=0),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=4),
                cell_creator=LinearVertexCellCurveCellCreator(
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_minmax),
                reconstruction_error_measure=reconstruction_error_measure_w
            )
        ],
        obera_iterations=obera_iterations
    )


def aero_qelvira_tem(smoothness_calculator=naive_piece_wise, refinement=1, obera_iterations=0, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            elvira_cc(angle_threshold=45),
            aero_q(angle_threshold=45),
            tem,
        ],
        obera_iterations=obera_iterations
    )


def aero_qtem(smoothness_calculator=naive_piece_wise, refinement=1, obera_iterations=0, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            aero_q(angle_threshold=45),
            tem,
        ],
        obera_iterations=obera_iterations
    )


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.paper_results_path,
        emissions_path=config.results_path,
        name='Corners',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )

    names_dict = {
        "aero_qelvira_vertex": "ELVIRA-WO + AEROS Quadratic + TEM + AEROS Vertex",
        "aero_qelvira_tem": "ELVIRA-WO + AEROS Quadratic + TEM",
        "aero_qvertex": "AEROS Quadratic + TEM + AEROS Vertex",
        "aero_qtem": "AEROS Quadratic + TEM",
        "obera_qtem": "OBERA Quadratic + TEM",
    }
    runsinfo.append_info(
        **{k.replace("_", "-"): v for k, v in names_dict.items()}
    )

    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "models",
        *map(fit_model, [
            aero_qelvira_vertex,
            aero_qelvira_tem,
            aero_qvertex,
            aero_qtem,
            # NamedPartial(aero_qtem, obera_iterations=500).add_sufix_to_name("_obera"),

            # piecewise_constant,
            # elvira,
            # elvira_w_oriented,
            #
            # linear_obera,
            # linear_obera_w,
            #
            # quadratic_obera_non_adaptive,
            quadratic_aero,
            #
            # quartic_aero,
        ]),
        recalculate=True
    )
    lab.execute(
        data_manager,
        num_cores=15,
        forget=False,
        save_on_iteration=None,
        num_cells_per_dim=[30],  # 20, 42, 84 168 , 84 4220,, 42
        image=[
            # "batata.jpg"
            # "yoda.jpg",
            # "DarthVader.jpeg",
            # "Ellipsoid_1680x1680.jpg",
            "ShapesVertex_1680x1680.jpg",
            # "HandVertex_1680x1680.jpg",
            # "Polygon_1680x1680.jpg",
        ],
        reconstruction_factor=[1],
    )

    winner_color_dict = {
        'PolynomialCelldegree (1, 1)': cgray,
        'CellCurveBaseCurveAveragePolynomialLine': cblue,
        "CellCurveBaseCurvePolynomialLine": cblue,
        'CellCurveBaseCurveAverageQuadraticCCQuadratic': cgreen,
        "CellCurveBaseCurveAveragePolynomialQuadratic": cgreen,
        "CellCurveBaseCurveVandermondePolynomialQuadratic": cgreen,
        "CellCurveBaseCurveAveragePolynomialPoly4": cpurple,
        'CellCurveBaseVertexLinearExtendedVertexCurvePolynomialByPartsLineaddCurvePolynomialByPartsLine': cred,
        'CellCurveBaseCurveVertexPolynomialVertexCurvePolynomialByPartsLineaddCurvePolynomialByPartsLine': corange,
        'CellCurveBaseVertexLinearExtendedVertexCurvePolynomialByPartsLineaddCurvePolynomialByPartsLineaddNoCurveRegionNoCurveRegion': corange
    }

    plot_reconstruction(
        data_manager,
        path=config.subcell_paper_figures_path,
        format=".pdf",
        plot_by=['image', 'models', 'num_cells_per_dim'],
        folder_by=["image"],
        axes_xy_proportions=(15, 15),
        difference=False,
        plot_curve=True,
        plot_curve_winner=False,
        plot_vh_classification=False,
        plot_singular_cells=False,
        alpha_true_image=1,
        alpha=0.65,
        plot_again=True,
        num_cores=1,
        num_cells_per_dim=[30],
        trim=((3, 3), (3, 3)),
        cmap=sns.color_palette("viridis", as_cmap=True),
        cmap_true_image=sns.color_palette("Greys_r", as_cmap=True),
        vmin=-1, vmax=1,
        labels=False,
        draw_mesh=False,
        numbers_on=False,
        axis_font_dict={},
        legend_font_dict={},
        xlabel=None,
        ylabel=None,
        xticks=None,
        yticks=None,
        # winner_color_dict=winner_color_dict
    )
