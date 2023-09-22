import time

import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import perplex_plot, one_line_iterator, generic_plot
from experiments.subcell_paper.global_params import EVALUATIONS
from experiments.subcell_paper.models2compare import aero_linear, aero_qelvira_vertex, elvira, quadratic, aero_lq, \
    winner_color_dict
from experiments.VizReconstructionUtils import plot_cells, plot_cells_identity, plot_cells_vh_classification_core, \
    plot_cells_not_regular_classification_core, plot_curve_core, draw_cell_borders
from experiments.subcell_paper.tools import calculate_averages_from_image, load_image, reconstruct, \
    get_reconstruction_error
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import CURVE_CELL_TYPE, REGULAR_CELL_TYPE, VERTEX_CELL_TYPE


def fit_model(sub_cell_model):
    def decorated_func(image, noise, num_cells_per_dim, reconstruction_factor, refinement, angle_threshold):
        image = load_image(image)
        avg_values = calculate_averages_from_image(image, num_cells_per_dim)
        np.random.seed(42)
        avg_values = avg_values + np.random.uniform(-noise, noise, size=avg_values.shape)

        model = sub_cell_model(refinement=refinement, angle_threshold=angle_threshold)

        t0 = time.time()
        model.fit(average_values=avg_values, indexer=ArrayIndexerNd(avg_values, "cyclic"))
        t_fit = time.time() - t0

        t0 = time.time()
        reconstruction = reconstruct(image, model.cells, model.resolution, reconstruction_factor,
                                     do_evaluations=EVALUATIONS)
        t_reconstruct = time.time() - t0

        return {
            "model": model,
            "time_to_fit": t_fit,
            "reconstruction_error": get_reconstruction_error(image, reconstruction=reconstruction,
                                                             reconstruction_factor=reconstruction_factor),
            "time_to_reconstruct": t_reconstruct
        }

    # need to change the name so the lab experiment saves the correct name and not the uniformly "decorated_func"
    # the other option is to pass to the block the name we wish to associate to the function.
    decorated_func.__name__ = sub_cell_model.__name__
    return decorated_func


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='AngleThreshold',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )

    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "models",
        *map(fit_model, [
            # elvira_oriented,
            elvira,
            aero_linear,
            # # aero_linear_oriented,
            quadratic,
            # # quadratic_oriented,
            # # aero_lq,
            aero_qelvira_vertex,
            # obera_aero_lq_vertex,
        ]),
        recalculate=False
    )
    lab.execute(
        data_manager,
        num_cores=15,
        forget=True,
        save_on_iteration=None,
        refinement=[1],
        num_cells_per_dim=[20, 42],  # 20, 42, 84 168 , 84
        noise=[0],
        image=[
            # "yoda.jpg",
            # "DarthVader.jpeg",
            # "Ellipsoid_1680x1680.jpg",
            "ShapesVertex_1680x1680.jpg",
            "HandVertex_1680x1680.jpg",
            # "Polygon_1680x1680.jpg",
        ],
        angle_threshold=[0, 20, 25, 30, 35, 40, 45],
        reconstruction_factor=[5]
    )

    generic_plot(
        data_manager,
        x="angle_threshold", y="reconstruction_error", label="models",
        plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
        # log="y",
        axes_by=["image"],
        plot_by=["num_cells_per_dim"]
    )

    generic_plot(
        data_manager,
        x="time_to_fit", y="reconstruction_error", label="models",
        plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
        # log="y",
        axes_by=["image"],
        plot_by=["num_cells_per_dim"]
    )
