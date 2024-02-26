import time

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from experiments.PaperPlots.exploring_methods_convergence import piecewise_constant, \
    quadratic_aero, quartic_aero, elvira, elvira_w_oriented, linear_obera, linear_obera_w, \
    quadratic_obera_non_adaptive
from experiments.global_params import EVALUATIONS, image_format
from experiments.tools import load_image, calculate_averages_from_image, reconstruct
from experiments.tools4binary_images import plot_reconstruction
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd


def fit_model(sub_cell_model):
    def decorated_func(image, num_cells_per_dim, reconstruction_factor):
        image = load_image(image)
        avg_values = calculate_averages_from_image(image, num_cells_per_dim)

        model = sub_cell_model()

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

    # need to change the name so the lab experiment saves the correct name and not the uniformly "decorated_func"
    # the other option is to pass to the block the name we wish to associate to the function.
    decorated_func.__name__ = sub_cell_model.__name__
    return decorated_func


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.paper_results_path,
        emissions_path=config.results_path,
        name='ImageReconstruction',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )

    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "models",
        *map(fit_model, [
            piecewise_constant,
            elvira,
            elvira_w_oriented,

            linear_obera,
            linear_obera_w,

            quadratic_obera_non_adaptive,
            quadratic_aero,

            quartic_aero,
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
            "batata.jpg"
            # "yoda.jpg",
            # "DarthVader.jpeg",
            # "Ellipsoid_1680x1680.jpg",
            # "ShapesVertex_1680x1680.jpg",
            # "HandVertex_1680x1680.jpg",
            # "Polygon_1680x1680.jpg",
        ],
        reconstruction_factor=[5],
    )

    plot_reconstruction(
        data_manager,
        path=config.subcell_paper_figures_path,
        format=image_format,
        plot_by=['image', 'models', 'num_cells_per_dim'],
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
        trim=((2, 3), (5, 5)),
        cmap="viridis",
        cmap_true_image="Greys_r",
        vmin=-1, vmax=1,
        labels=False,
        draw_mesh=False,
        numbers_on=False,
        axis_font_dict={},
        legend_font_dict={},
        xlabel=None,
        ylabel=None,
        xticks=None,
        yticks=None
    )
