import operator
import time
from functools import partial

import numpy as np

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import ClassPartialInit
from PerplexityLab.visualization import perplex_plot, one_line_iterator
from experiments.VizReconstructionUtils import plot_cells, plot_cells_identity, plot_cells_vh_classification_core, \
    plot_cells_not_regular_classification_core, plot_curve_core, draw_cell_borders
from experiments.subcell_paper.tools import calculate_averages_from_image, load_image
from experiments.subcell_paper.global_params import CCExtraWeight, CurveAverageQuadraticCC
from experiments.subcell_paper.obera_experiments import get_sub_cell_model
from lib.AuxiliaryStructures.Constants import REGULAR_CELL, CURVE_CELL, VERTEX_CELL
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import CURVE_CELL_TYPE, REGULAR_CELL_TYPE, VERTEX_CELL_TYPE
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_regular_cells, \
    get_opposite_regular_cells_by_stencil
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesCurveCellCreator
from lib.CellCreators.CurveCellCreators.VertexCellCreator import LinearVertexCellCurveCellCreator
from lib.CellCreators.RegularCellCreator import PiecewiseConstantRegularCellCreator
from lib.CellCreators.VertexCellCreators.VertexCellCreatorBase import VertexCellCreatorUsingNeighboursLines
from lib.CellIterators import iterate_by_condition_on_smoothness, iterate_by_reconstruction_error_and_smoothness
from lib.CellOrientators import BaseOrientator, OrientByGradient, OrientPredefined
from lib.Curves.AverageCurves import CurveAveragePolynomial
from lib.SmoothnessCalculators import naive_piece_wise
from lib.StencilCreators import StencilCreatorAdaptive, StencilCreatorFixedShape
from lib.SubCellReconstruction import SubCellReconstruction, ReconstructionErrorMeasure, CellCreatorPipeline

EVALUATIONS = True


def fit_model(sub_cell_model):
    def decorated_func(image, noise, num_cells_per_dim, reconstruction_factor, iterations):
        image = load_image(image)
        avg_values = calculate_averages_from_image(image, num_cells_per_dim)
        np.random.seed(42)
        avg_values = avg_values + np.random.uniform(-noise, noise, size=avg_values.shape)

        model = sub_cell_model(iterations)

        t0 = time.time()
        model.fit(average_values=avg_values, indexer=ArrayIndexerNd(avg_values, "cyclic"))
        t_fit = time.time() - t0

        t0 = time.time()
        if EVALUATIONS:
            reconstruction = model.reconstruct_arbitrary_size(np.array(np.shape(image)) // reconstruction_factor)
        else:
            reconstruction = model.reconstruct_by_factor(
                resolution_factor=np.array(np.array(np.shape(image)) / np.array(model.resolution), dtype=int))
        t_reconstruct = time.time() - t0

        # if reconstruction_factor > 1:
        #     if EVALUATIONS:
        #         step = np.array(np.array(np.shape(image)) / np.array(np.shape(reconstruction)), dtype=int)
        #         image = image[np.arange(0, np.shape(image)[0], step[0], dtype=int)][:,
        #                 np.arange(0, np.shape(image)[1], step[1], dtype=int)]
        #     else:
        #         image = calculate_averages_from_image(image, num_cells_per_dim=np.shape(reconstruction))
        # reconstruction_error = np.abs(np.array(reconstruction) - image)

        # t0 = time.time()
        # reconstruction = model.reconstruct_by_factor(resolution_factor=sub_discretization2bound_error)
        # t_reconstruct = time.time() - t0
        # # if refinement is set in place
        # reconstruction = calculate_averages_from_image(reconstruction, tuple(
        #     (np.array(np.shape(image)) * sub_discretization2bound_error).tolist()))

        return {
            "model": model,
            "time_to_fit": t_fit,
            "reconstruction": reconstruction,
            # "reconstruction_error": reconstruction_error,
            "time_to_reconstruct": t_reconstruct
        }

    # need to change the name so the lab experiment saves the correct name and not the uniformly "decorated_func"
    # the other option is to pass to the block the name we wish to associate to the function.
    decorated_func.__name__ = sub_cell_model.__name__
    return decorated_func


@fit_model
def quadratic_vertex(iterations):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=naive_piece_wise,
        reconstruction_error_measure=ReconstructionErrorMeasure(StencilCreatorFixedShape((3, 3)),
                                                                metric=2,
                                                                central_cell_extra_weight=100),
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_condition_on_smoothness, value=REGULAR_CELL,
                                      condition=operator.eq),  # only regular cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
                cell_creator=PiecewiseConstantRegularCellCreator(
                    apriori_up_value=1, apriori_down_value=0, dimensionality=2)
            ),
            # ------------ ELVIRA ------------ #
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2),
                stencil_creator=StencilCreatorFixedShape((3, 3)),
                cell_creator=ELVIRACurveCellCreator(
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil)),
            # ------------ TEM ------------ #
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientPredefined(predefined_axis=0),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(3, 3)),
                cell_creator=VertexCellCreatorUsingNeighboursLines(
                    regular_opposite_cell_searcher=partial(get_opposite_regular_cells, direction="grad"))
            ),
            # ------------ AERO Quadratic ------------ #
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientPredefined(predefined_axis=0),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=3),
                cell_creator=ValuesCurveCellCreator(
                    vander_curve=CurveAverageQuadraticCC,
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil)),
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientPredefined(predefined_axis=1),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=3),
                cell_creator=ValuesCurveCellCreator(
                    vander_curve=CurveAverageQuadraticCC,
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil)),
            # ------------ AVRO ------------ #
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientPredefined(predefined_axis=0),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=4),
                cell_creator=LinearVertexCellCurveCellCreator(
                    regular_opposite_cell_searcher=get_opposite_regular_cells)
            ),
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientPredefined(predefined_axis=1),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=4),
                cell_creator=LinearVertexCellCurveCellCreator(
                    regular_opposite_cell_searcher=get_opposite_regular_cells)
            ),
        ],
        obera_iterations=iterations
    )


@perplex_plot()
@one_line_iterator
def plot_reconstruction(fig, ax, image, num_cells_per_dim, model, reconstruction, alpha=0.5,
                        plot_original_image=True,
                        difference=False, plot_curve=True, plot_curve_winner=False, plot_vh_classification=True,
                        plot_singular_cells=True, cmap="magma", trim=((0, 0), (0, 0)), numbers_on=True):
    model_resolution = np.array(model.resolution)
    image = load_image(image)

    if plot_original_image:
        plot_cells(ax, colors=image, mesh_shape=model_resolution, alpha=alpha, cmap="Greys_r",
                   vmin=np.min(image), vmax=np.max(image))

    if difference:
        # TODO: should be the evaluations not the averages.
        image = calculate_averages_from_image(image, num_cells_per_dim=np.shape(reconstruction))
        plot_cells(ax, colors=reconstruction - image, mesh_shape=model.resolution, alpha=alpha, cmap=cmap, vmin=-1,
                   vmax=1)
    else:
        plot_cells(ax, colors=reconstruction, mesh_shape=model.resolution, alpha=alpha, cmap=cmap, vmin=-1, vmax=1)

    if plot_curve:
        if plot_curve_winner:
            plot_cells_identity(ax, model.resolution, model.cells, alpha=0.8)
            # plot_cells_type_of_curve_core(ax, model.resolution, model.cells, alpha=0.8)
        elif plot_vh_classification:
            plot_cells_vh_classification_core(ax, model.resolution, model.cells, alpha=0.8)
        elif plot_singular_cells:
            plot_cells_not_regular_classification_core(ax, model.resolution, model.cells, alpha=0.8)
        plot_curve_core(ax, curve_cells=[cell for cell in model.cells.values() if
                                         cell.CELL_TYPE != REGULAR_CELL_TYPE])

    draw_cell_borders(
        ax, mesh_shape=num_cells_per_dim,
        refinement=model_resolution // num_cells_per_dim,
        numbers_on=numbers_on,
        prop_ticks=10 / num_cells_per_dim  # each 10 cells a tick
    )
    ax.set_xlim((-0.5 + trim[0][0], model.resolution[0] - trim[0][1] - 0.5))
    ax.set_ylim((model.resolution[1] - trim[1][0] - 0.5, trim[1][1] - 0.5))


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='Vertices',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )

    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "models",
        quadratic_vertex,
        recalculate=False
    )
    lab.execute(
        data_manager,
        num_cores=15,
        forget=False,
        save_on_iteration=None,
        refinement=[1],
        num_cells_per_dim=[20, 42, 84],  # 20, 42, 84 168
        noise=[0],
        image=[
            "yoda.jpg",
            "DarthVader.jpeg",
            "ShapesVertex_1680x1680.jpg",
            "HandVertex_1680x1680.jpg",
            "Polygon_1680x1680.jpg",
        ],
        iterations=[0],  # 500
        reconstruction_factor=[5],
    )

    for model in data_manager["model"]:
        print(f"{model}: "
              f"#CURVES={sum([1 for cell in model.cells.values() if cell.CELL_TYPE == CURVE_CELL_TYPE])}"
              f"#VERTICES={sum([1 for cell in model.cells.values() if cell.CELL_TYPE == VERTEX_CELL_TYPE])}", )

    plot_reconstruction(
        data_manager,
        name="Reconstruction",
        folder='Reconstruction',
        plot_by=['image', 'models', 'num_cells_per_dim', "iterations"],
        axes_xy_proportions=(15, 15),
        difference=False,
        plot_curve=True,
        plot_curve_winner=False,
        plot_vh_classification=False,
        plot_singular_cells=False,
        plot_original_image=True,
        numbers_on=True,
        plot_again=True,
        num_cores=1,
    )
