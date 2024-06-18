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
from experiments.PaperPlots.paper_corners_plot import aero_qelvira_vertex, reconstruction_error_measure_default, \
    piecewise01, elvira_cc

from experiments.PaperPlots.paper_smooth_domains_plot import fit_model
from experiments.PaperPlots.exploring_methods_convergence import quadratic_aero
from experiments.global_params import CurveAverageQuadraticCC, CCExtraWeight, cgray, cblue, cgreen, cred, \
    corange, cpurple, runsinfo, image_format
from experiments.tools import curve_cells_fitting_times
from experiments.tools4binary_images import plot_reconstruction

from lib.AuxiliaryStructures.Constants import REGULAR_CELL, CURVE_CELL
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_regular_cells_by_minmax
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesCurveCellCreator
from lib.CellCreators.CurveCellCreators.VertexCellCreator import LinearVertexCellCurveCellCreator
from lib.CellCreators.RegularCellCreator import PiecewiseConstantRegularCellCreator, MirrorCellCreator
from lib.CellCreators.VertexCellCreators.VertexCellCreatorBase import VertexCellCreatorUsingNeighboursLines
from lib.CellIterators import iterate_by_reconstruction_error_and_smoothness, \
    iterate_by_condition_on_smoothness, iterate_all
from lib.CellOrientators import BaseOrientator, OrientByGradient, OrientPredefined
from lib.SmoothnessCalculators import naive_piece_wise, indifferent
from lib.StencilCreators import StencilCreatorFixedShape, StencilCreatorAdaptive
from lib.SubCellReconstruction import ReconstructionErrorMeasure, ReconstructionErrorMeasureDefaultStencil, \
    CellCreatorPipeline, SubCellReconstruction

# cmap = "viridis"
alpha = 1
cmap = "coolwarm"
vmin = 0
vmax = 1


def original_data(smoothness_calculator=naive_piece_wise, refinement=1, obera_iterations=0, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=indifferent,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            CellCreatorPipeline(
                cell_iterator=iterate_all,  # only regular cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
                cell_creator=MirrorCellCreator(dimensionality=2)
            ),
        ],
        obera_iterations=obera_iterations
    )


def elvira(smoothness_calculator=naive_piece_wise, refinement=1, obera_iterations=0, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            elvira_cc(angle_threshold=45)
        ],
        obera_iterations=obera_iterations
    )


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.subcell_presentation_path,
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
            aero_qelvira_vertex,
            elvira,
            original_data
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

    plot_reconstruction(
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
        alpha_true_image=0,
        alpha=alpha,
        plot_again=True,
        num_cores=1,
        num_cells_per_dim=[30],
        trim=((3, 3), (3, 3)),
        cmap=cmap,
        cmap_true_image=sns.color_palette("Greys_r", as_cmap=True),
        vmin=vmin, vmax=vmax,
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
