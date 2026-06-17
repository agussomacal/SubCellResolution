import warnings

import matplotlib
import seaborn as sns

import matplotlib.pylab as plt

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.visualization import generic_plot, make_data_frames, save_fig
from experiments.OtherExperiments.SubcellExperiments.models2compare import elvira, winner_color_dict, \
    qelvira, quadratic
from experiments.global_params import image_format, cred
from experiments.tools import get_reconstruction_error, load_image
from experiments.tools4binary_images import fit_model, plx_plot_reconstruction4img

if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='Refinement',
        format=JOBLIB,
        trackCO2=False,
        country_alpha_code="FR"
    )

    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "models",
        *map(fit_model, [
            # elvira,
            # aero_lq,
            # qelvira,
            quadratic,
        ]),
        recalculate=False
    )
    with warnings.catch_warnings(action="ignore"):
        lab.execute(
            data_manager,
            num_cores=1,
            forget=False,
            save_on_iteration=1,
            refinement=[1, 2, 3],
            num_cells_per_dim=[20, 42],  # 20, 42, 84 168 , 84
            noise=[0],
            image=[
                "batata.jpg",
                "ShapesVertex.jpg",
                # "yoda.jpg",
                # "DarthVader.jpeg",
                # "Ellipsoid_1680x1680.jpg",
                # "ShapesVertex_1680x1680.jpg",
                # "HandVertex_1680x1680.jpg",
                # "Polygon_1680x1680.jpg",
            ],
            reconstruction_factor=[1],
            angle_threshold=[0]
        )

    matplotlib.rcParams['text.usetex'] = False
    curve_color = cred
    cmap_reconstruction = "Reds"
    cmap_true_image = "Greys_r"
    plx_plot_reconstruction4img(
        data_manager,
        path=data_manager.path,
        format=image_format,
        plot_by=['image', 'num_cells_per_dim', 'models', "refinement"],
        axes_xy_proportions=(15, 15),
        difference=False,
        plot_curve=True,
        plot_curve_winner=False,
        plot_vh_classification=False,
        plot_singular_cells=False,
        alpha_true_image=0.15,
        alpha=0,
        plot_again=True,
        num_cores=1,
        # num_cells_per_dim=[15],
        trim=((1, 1), (2, 2)),
        cmap=cmap_reconstruction,
        cmap_true_image=cmap_true_image,
        curve_color=curve_color,
        vmin=0, vmax=1,
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

    # df = next(make_data_frames(
    #     data_manager,
    #     var_names=["models", "refinement", "image", "num_cells_per_dim", "N", "reconstruction_error"],
    #     group_by=["models"],
    #     N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
    #     reconstruction_error=lambda image, reconstruction, reconstruction_factor: get_reconstruction_error(
    #         load_image(image),
    #         reconstruction,
    #         reconstruction_factor),
    # ))[1]

    # generic_plot(
    #     data_manager,
    #     x="N", y="reconstruction_error", label="model_name",
    #     N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
    #     model_name=lambda models, refinement: f"{models} ref: {refinement}",
    #     plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
    #     # log="y",
    #     reconstruction_error=lambda image, reconstruction, reconstruction_factor: get_reconstruction_error(
    #         load_image(image),
    #         reconstruction,
    #         reconstruction_factor),
    #     axes_by=["image"],
    #     plot_by=["num_cells_per_dim"]
    # )
    #
    # plot_reconstruction4img(
    #     data_manager,
    #     path=config.subcell_paper_figures_path,
    #     name="Reconstruction",
    #     folder='Reconstruction',
    #     axes_by=[],
    #     plot_by=['image', 'models', 'num_cells_per_dim', "refinement"],
    #     models=["elvira", "qelvira"],
    #     image=[
    #         "ShapesVertex_1680x1680.jpg",
    #         "HandVertex_1680x1680.jpg",
    #     ],
    #     axes_xy_proportions=(15, 15),
    #     difference=False,
    #     plot_curve=True,
    #     plot_curve_winner=False,
    #     plot_vh_classification=False,
    #     plot_singular_cells=False,
    #     plot_original_image=True,
    #     numbers_on=True,
    #     plot_again=True,
    #     num_cores=15,
    #     # cmap="YlOrBr",
    #     # cmapoi="YlGn",
    #     # alphaio=0.7,
    #     # alpha=0,
    #     # winner_color_dict=winner_color_dict,
    #     format=".pdf",
    #     draw_cells=False
    # )
    #
    # plot_reconstruction4img(
    #     data_manager,
    #     name="Reconstruction",
    #     folder='Reconstruction',
    #     plot_by=['image', 'models', 'num_cells_per_dim', "refinement"],
    #     # models=[
    #     #     "aero_lq_vertex"
    #     # ],
    #     # image=[
    #     #     # "yoda.jpg",
    #     #     # "DarthVader.jpeg",
    #     #     # "Ellipsoid_1680x1680.jpg",
    #     #     "ShapesVertex_1680x1680.jpg",
    #     #     # "HandVertex_1680x1680.jpg",
    #     #     # "Polygon_1680x1680.jpg",
    #     # ],
    #     axes_xy_proportions=(15, 15),
    #     difference=False,
    #     plot_curve=True,
    #     plot_curve_winner=False,
    #     plot_vh_classification=False,
    #     plot_singular_cells=False,
    #     plot_original_image=True,
    #     numbers_on=True,
    #     plot_again=True,
    #     num_cores=15,
    #     curve_color=winner_color_dict,
    #     cmap="plasma"
    # )
