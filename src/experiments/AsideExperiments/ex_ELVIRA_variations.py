import time

import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import generic_plot
from experiments.subcell_paper.global_params import EVALUATIONS
from experiments.subcell_paper.models2compare import aero_linear, aero_lq_vertex, quadratic, winner_color_dict, \
    reconstruction_error_measure_default, piecewise01, elvira_cc
from experiments.subcell_paper.tools import load_image, calculate_averages_from_image, reconstruct, \
    get_reconstruction_error
from experiments.subcell_paper.tools4binary_images import fit_model, plot_reconstruction
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.SmoothnessCalculators import naive_piece_wise
from lib.SubCellReconstruction import SubCellReconstruction


def experiment(image, noise, num_cells_per_dim, reconstruction_factor, refinement, angle_threshold, weight):
    image = load_image(image)
    avg_values = calculate_averages_from_image(image, num_cells_per_dim)
    np.random.seed(42)
    avg_values = avg_values + np.random.uniform(-noise, noise, size=avg_values.shape)

    model = SubCellReconstruction(
        name="All",
        smoothness_calculator=naive_piece_wise,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            elvira_cc(angle_threshold=angle_threshold, weight=weight)
        ],
        obera_iterations=0
    )

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
        "reconstruction": reconstruction,
        "time_to_reconstruct": t_reconstruct
    }


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='ELVIRAVariations',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )

    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "elvira_variations",
        experiment,
        recalculate=False
    )
    lab.execute(
        data_manager,
        num_cores=15,
        forget=False,
        save_on_iteration=None,
        refinement=[1],
        num_cells_per_dim=[30],  # 20, 42, 84 168 , 84 4220,, 42
        noise=[0],
        image=[
            "Ellipsoid_1680x1680.jpg",
            "ShapesVertex_1680x1680.jpg",
            "HandVertex_1680x1680.jpg",
            "Polygon_1680x1680.jpg",
        ],
        reconstruction_factor=[6],
        angle_threshold=[0, 25, 30, 45],
        weight=[0, 1, 2, 3, 4, 5, 20, 50, 100, 500, 1000],
    )

    generic_plot(
        data_manager,
        x="weight", y="error", label="angle_threshold",
        error=lambda image, reconstruction, reconstruction_factor:
        get_reconstruction_error(load_image(image), reconstruction, reconstruction_factor),
        plot_func=NamedPartial(sns.lineplot, marker="o", palette="winter"),
        axes_by=['image'],
        log="x"
    )

    generic_plot(
        data_manager,
        x="weight", y="time_to_fit", label="angle_threshold",
        error=lambda image, reconstruction, reconstruction_factor:
        get_reconstruction_error(load_image(image), reconstruction, reconstruction_factor),
        plot_func=NamedPartial(sns.lineplot, marker="o", palette="winter"),
        axes_by=['image'],
        log="x"
    )

    plot_reconstruction(
        data_manager,
        path=config.subcell_paper_figures_path,
        name="Reconstruction",
        folder='Reconstruction',
        axes_by=["weight"],
        plot_by=['image', 'angle_threshold', 'num_cells_per_dim'],
        axes_xy_proportions=(15, 15),
        difference=False,
        plot_curve=True,
        plot_curve_winner=False,
        plot_vh_classification=False,
        plot_singular_cells=False,
        plot_original_image=True,
        numbers_on=True,
        plot_again=True,
        num_cores=15,
        # cmap="YlOrBr",
        # cmapoi="YlGn",
        # alphaio=0.7,
        # alpha=0,
        # format=".pdf"
    )
