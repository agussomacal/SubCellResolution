import time
from functools import partial

import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd

import config
from experiments.TaylorEffect.taylor_effect import enhance_image, image_reconstruction
from experiments.VizReconstructionUtils import plot_cells, draw_cell_borders
from experiments.image_reconstruction import plot_reconstruction
from experiments.models import calculate_averages_from_image
from experiments.subcell_paper.function_families import load_image
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_cells_by_grad
from lib.CellCreators.RegularCellCreator import PolynomialRegularCellCreator, weight_cells, weight_cells_extra_weight, \
    weight_cells_by_smoothness
from lib.CellIterators import iterate_all, iterate_by_smoothness
from lib.CellOrientators import BaseOrientator, OrientByGradient
from lib.SmoothnessCalculators import indifferent, by_gradient
from lib.StencilCreators import StencilCreatorSameRegionAdaptive, StencilCreatorFixedShape, \
    StencilCreatorSmoothnessDistTradeOff
from lib.SubCellReconstruction import CellCreatorPipeline, SubCellReconstruction, ReconstructionErrorMeasureBase, \
    ReconstructionErrorMeasure
from src.DataManager import DataManager, JOBLIB
from src.Indexers import ArrayIndexerNd
from src.LaTexReports import Code2LatexConnector
from src.LabPipeline import LabPipeline
from src.viz_utils import perplex_plot, generic_plot, test_plot


def core_experiment(enhanced_image, num_cells_per_dim, noise, dist_trade_off, central_cell_importance, delta, epsilon,
                    degree=2, reduced_image_size_factor=1):
    model = SubCellReconstruction(
        name="Polynomial",
        smoothness_calculator=by_gradient,
        reconstruction_error_measure=ReconstructionErrorMeasure(
            stencil_creator=StencilCreatorFixedShape((3, 3)), central_cell_extra_weight=1, metric="l2"),
        refinement=1,
        cell_creators=
        [
            # regular cell with quadratics
            CellCreatorPipeline(
                cell_iterator=iterate_all,  # all cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorSmoothnessDistTradeOff(stencil_shape=(3, 3),
                                                                     dist_trade_off=dist_trade_off),
                cell_creator=PolynomialRegularCellCreator(
                    degree=degree, dimensionality=2, noisy=False, full_rank=False,
                    weight_function=partial(weight_cells_by_smoothness,
                                            central_cell_importance=central_cell_importance,
                                            delta=delta, epsilon=epsilon)
                )
            )
        ]
    )

    np.random.seed(42)
    avg_values = calculate_averages_from_image(enhanced_image, num_cells_per_dim)
    avg_values += np.random.uniform(-noise, noise, size=avg_values.shape)

    t0 = time.time()
    model.fit(average_values=avg_values,
              indexer=ArrayIndexerNd(avg_values, "cyclic"))
    t_fit = time.time() - t0

    enhanced_image = calculate_averages_from_image(enhanced_image,
                                                   np.array(np.shape(enhanced_image)) // reduced_image_size_factor)
    t0 = time.time()
    reconstruction = model.reconstruct_arbitrary_size(np.shape(enhanced_image))
    t_reconstruct = time.time() - t0
    return t_fit, t_reconstruct, reconstruction, model


def experiment(image, amplitude, num_cells_per_dim, noise, dist_trade_off, central_cell_importance, delta, epsilon,
               degree=2, reduced_image_size_factor=6):
    enhanced_image = enhance_image(image, amplitude)["enhanced_image"]
    image = load_image(image)
    t_fit, t_reconstruct, reconstruction, _ = core_experiment(enhanced_image, num_cells_per_dim, noise, dist_trade_off,
                                                              central_cell_importance, delta, epsilon,
                                                              degree, reduced_image_size_factor)
    reduced_shape = np.array(np.shape(enhanced_image)) // reduced_image_size_factor
    resolution_factor = reduced_shape // num_cells_per_dim

    enhanced_image = calculate_averages_from_image(enhanced_image, reduced_shape)

    image = calculate_averages_from_image(image, num_cells_per_dim)
    edge_mask = (1 > image) & (image > 0)  # cells with an edge passing through
    # extend repeating mask to have the same shape as reconstructed images
    edge_mask = np.repeat(np.repeat(edge_mask, resolution_factor[0], axis=0), resolution_factor[1], axis=1)

    # only test error in the regular regions
    return {
        "time_to_fit": t_fit,
        "time_to_reconstruct": t_reconstruct,
        "mse": np.sqrt(np.mean((reconstruction - enhanced_image)[~edge_mask] ** 2))  # filter edges from error
    }


@test_plot
def plot_reconstruction(fig, ax, image, amplitude, num_cells_per_dim, noise, dist_trade_off, central_cell_importance,
                        delta, epsilon, degree, alpha=0.5, plot_original_image=True, difference=False, cmap="magma",
                        trim=((0, 0), (0, 0)), numbers_on=True, reduced_image_size_factor=1, *args, **kwargs):
    image = enhance_image(image, amplitude)["enhanced_image"]
    _, _, reconstruction, model = core_experiment(
        image, num_cells_per_dim, noise, dist_trade_off, central_cell_importance, delta, epsilon, degree,
        reduced_image_size_factor)

    model_resolution = np.array(model.resolution)

    image = calculate_averages_from_image(image, np.array(np.shape(image)) // reduced_image_size_factor)
    if plot_original_image:
        plot_cells(ax, colors=image, mesh_shape=model_resolution, alpha=alpha, cmap="Greys_r",
                   vmin=np.min(image), vmax=np.max(image))

    if difference:
        plot_cells(ax, colors=reconstruction - image, mesh_shape=model_resolution, alpha=alpha, cmap=cmap, vmin=-1,
                   vmax=1)
    else:
        plot_cells(ax, colors=reconstruction, mesh_shape=model_resolution, alpha=alpha, cmap=cmap, vmin=-1, vmax=1)

    draw_cell_borders(
        ax, mesh_shape=num_cells_per_dim,
        refinement=model_resolution // num_cells_per_dim,
        numbers_on=numbers_on,
        prop_ticks=10 / num_cells_per_dim  # each 10 cells a tick
    )
    ax.set_xlim((-0.5 + trim[0][0], model.resolution[0] - trim[0][1] - 0.5))
    ax.set_ylim((model.resolution[1] - trim[1][0] - 0.5, trim[1][1] - 0.5))


if __name__ == "__main__":
    report = Code2LatexConnector(path=config.subcell_paper_path, filename='main')

    data_manager = DataManager(
        path=config.results_path,
        name='RegularFitDistTradeOff',
        format=JOBLIB
    )
    data_manager.load()

    lab = LabPipeline()

    lab.define_new_block_of_functions(
        "regular_error",
        experiment
    )

    lab.execute(
        data_manager,
        num_cores=3,
        recalculate=False,
        forget=False,
        save_on_iteration=None,
        degree=[2],
        dist_trade_off=[0, 0.25, 0.5, 0.75, 1],
        central_cell_importance=[0],
        # delta=[0, 0.05, 0.5],
        delta=[0.5],
        # epsilon=[1e-5, 1e5],
        epsilon=[np.inf],
        num_cells_per_dim=[
            28, 28 * 2  # , 28 * 4
        ],
        noise=[0],
        amplitude=[0, 1e-3, 1e-2, 1e-1, 5e-1, 1],
        reduced_image_size_factor=[6],
        image=[
            "Elipsoid_1680x1680.jpg"
        ]
    )

    # plot_reconstruction(data_manager.path, image="Elipsoid_1680x1680.jpg", amplitude=5e-1, num_cells_per_dim=28,
    #                     noise=0, dist_trade_off=1,
    #                     central_cell_importance=0, delta=0.5, epsilon=np.inf, degree=1,
    #                     alpha=0.5, plot_original_image=True, difference=False, cmap="magma",
    #                     trim=((0, 0), (0, 0)), numbers_on=True, reduced_image_size_factor=6)

    generic_plot(data_manager, path=report.get_plot_path(), x="amplitude", y="mse", label="dist_trade_off",
                 plot_fun=sns.lineplot,
                 other_plot_funcs=(), log="y",
                 plot_by=["num_cells_per_dim"])

    # =============== =============== =============== #
    # =============== =============== =============== #
    # dist_trade_off = 0.5
    data_manager = DataManager(
        path=config.results_path,
        name='RegularFitWeights',
        format=JOBLIB
    )
    data_manager.load()

    lab = LabPipeline()

    lab.define_new_block_of_functions(
        "regular_error",
        experiment
    )

    lab.execute(
        data_manager,
        num_cores=3,
        recalculate=False,
        forget=False,
        save_on_iteration=None,
        degree=[2],
        dist_trade_off=[0.5],
        central_cell_importance=[0],
        # delta=[0, 0.05, 0.5],
        delta=[0, 0.05, 0.5],
        # epsilon=[1e-5, 1e5],
        epsilon=[1e-5, np.inf],
        num_cells_per_dim=[
            28, 28 * 2  # , 28 * 4
        ],
        noise=[0],
        amplitude=[0, 1e-3, 1e-2, 1e-1, 5e-1, 1],
        reduced_image_size_factor=[6],
        image=[
            "Elipsoid_1680x1680.jpg"
        ]
    )

    generic_plot(data_manager, path=report.get_plot_path(), x="amplitude", y="mse", label="delta",
                 plot_fun=sns.lineplot,
                 other_plot_funcs=(), log="y",
                 plot_by=["num_cells_per_dim", "epsilon"])

    # =============== =============== =============== #
    # =============== =============== =============== #
    # dist_trade_off = 0.5
    # delta = 0.05
    # epsilon = 1e-5

    data_manager = DataManager(
        path=config.results_path,
        name='RegularFitWeightsCCImportance',
        format=JOBLIB
    )
    data_manager.load()

    lab = LabPipeline()

    lab.define_new_block_of_functions(
        "regular_error",
        experiment
    )

    lab.execute(
        data_manager,
        num_cores=3,
        recalculate=False,
        forget=False,
        save_on_iteration=None,
        degree=[2],
        dist_trade_off=[0.5],
        central_cell_importance=[0, 1, 2, 3, 6, 12, 25, 50, 100],
        # delta=[0, 0.05, 0.5],
        delta=[0.05],
        # epsilon=[1e-5, 1e5],
        epsilon=[1e-5],
        num_cells_per_dim=[
            28, 28 * 2  # , 28 * 4
        ],
        noise=[0],
        amplitude=[0, 1e-3, 1e-2, 1e-1, 5e-1, 1],
        reduced_image_size_factor=[6],
        image=[
            "Elipsoid_1680x1680.jpg"
        ]
    )

    generic_plot(data_manager, path=report.get_plot_path(), x="amplitude", y="mse", label="central_cell_importance",
                 plot_fun=sns.lineplot,
                 other_plot_funcs=(), log="y",
                 plot_by=["num_cells_per_dim", "epsilon"])

    # central_cell_importance = 100

    report.compile()
