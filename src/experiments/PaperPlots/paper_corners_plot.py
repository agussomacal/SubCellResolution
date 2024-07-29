import operator
from functools import partial
from itertools import chain

import numpy as np
import seaborn as sns

import config

from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import copy_main_script_version
from PerplexityLab.visualization import make_data_frames

from experiments.PaperPlots.paper_smooth_domains_plot import fit_model
from experiments.PaperPlots.exploring_methods_convergence import quadratic_aero, elvira, elvira_w_oriented
from experiments.global_params import CurveAverageQuadraticCC, CCExtraWeight, cgray, cblue, cgreen, cred, \
    corange, cpurple, runsinfo, image_format
from experiments.tools import curve_cells_fitting_times
from experiments.tools4binary_images import plot_reconstruction4img

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


tem = lambda reconstruction_error_measure: CellCreatorPipeline(
    cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                          condition=operator.eq),
    orientator=OrientPredefined(predefined_axis=0),
    stencil_creator=StencilCreatorFixedShape(stencil_shape=(3, 3)),
    cell_creator=VertexCellCreatorUsingNeighboursLines(
        regular_opposite_cell_searcher=get_opposite_regular_cells_by_minmax,
    ),
    reconstruction_error_measure=reconstruction_error_measure
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


def elvira_vertex(smoothness_calculator=naive_piece_wise, refinement=1, obera_iterations=0, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            elvira_cc(angle_threshold=45),
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

# rem = ReconstructionErrorMeasureDefaultStencil(
#                     StencilCreatorFixedShape((3, 3)),
#                     metric=2, central_cell_extra_weight=10)

# def aero_qelvira_vertex(smoothness_calculator=naive_piece_wise, refinement=1, obera_iterations=0, *args, **kwargs):
#     return SubCellReconstruction(
#         name="All",
#         smoothness_calculator=smoothness_calculator,
#         reconstruction_error_measure=reconstruction_error_measure_default,
#         refinement=refinement,
#         cell_creators=
#         [
#             piecewise01,
#             elvira_cc(angle_threshold=0, reconstruction_error_measure=rem),
#             aero_q(angle_threshold=0, reconstruction_error_measure=rem),
#             tem(reconstruction_error_measure=rem),
#             # ------------ AVRO ------------ #
#             CellCreatorPipeline(
#                 cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
#                                       condition=operator.eq),
#                 orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2, method="sobel",
#                                             angle_threshold=0),
#                 stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=4),
#                 cell_creator=LinearVertexCellCurveCellCreator(
#                     regular_opposite_cell_searcher=get_opposite_regular_cells_by_minmax),
#                 reconstruction_error_measure=rem
#             ),
#         ],
#         obera_iterations=obera_iterations,
#         eps_complexity=[0, 0, 0, 0, 0],
#     )


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
            tem(reconstruction_error_measure_w),
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

names_dict = {
    "aero_qelvira_vertex": "ELVIRA-WO + AEROS Quadratic + TEM + AEROS Vertex",
    "elvira_vertex": "ELVIRA + TEM + AEROS Vertex",
    "aero_qelvira_tem": "ELVIRA-WO + AEROS Quadratic + TEM",
    "aero_qvertex": "AEROS Quadratic + TEM + AEROS Vertex",
    "aero_qtem": "AEROS Quadratic + TEM",
    "obera_qtem": "OBERA Quadratic + TEM",
    "elvira": "ELVIRA",
    "elvira_w_oriented": "ELVIRA-WO",
}
runsinfo.append_info(
    **{k.replace("_", "-"): v for k, v in names_dict.items()}
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

    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "models",
        *map(fit_model, [
            elvira,
            elvira_w_oriented,
            quadratic_aero,
            aero_qvertex,
            aero_qtem,
            elvira_vertex,
            aero_qelvira_vertex,
        ]),
        recalculate=False
    )
    lab.execute(
        data_manager,
        num_cores=15,
        forget=False,
        save_on_iteration=None,
        num_cells_per_dim=[30],  # 20, 42, 84 168 , 84 4220,, 42
        image=[
            # "batata.jpg",
            "ShapesVertex.jpg",
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

    plot_reconstruction4img(
        data_manager,
        path=config.subcell_paper_figures_path,
        format=image_format,
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
    )

    # ========== =========== ========== =========== #
    #               Experiment Times                #
    # ========== =========== ========== =========== #
    # times to fit cell
    df = next(make_data_frames(
        data_manager,
        var_names=["models", "time"],
        group_by=[],
        time=curve_cells_fitting_times,
    ))[1].groupby("models").apply(lambda x: np.nanmean(list(chain(*x["time"].values.tolist()))))
    runsinfo.append_info(
        **{"corners-" + k.replace("_", "-") + "-time": f"{v:.1g}" for k, v in df.items()}
    )

    print("CO2 consumption: ", data_manager.CO2kg)
    copy_main_script_version(__file__, data_manager.path)
