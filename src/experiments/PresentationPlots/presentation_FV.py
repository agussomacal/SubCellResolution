from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.miscellaneous import NamedPartial, copy_main_script_version
from PerplexityLab.visualization import generic_plot, LegendOutsidePlot, perplex_plot, one_line_iterator
from experiments.PaperPlots.exploring_scheme_methods import scheme_reconstruction_error
from experiments.PaperPlots.paper_scheme_plot_transp import plot_reconstruction_init_final, model_color, names_dict, \
    num_cells_per_dim
from experiments.PresentationPlots.presentation_convergence_plot import extra_process4plot
from experiments.VizReconstructionUtils import plot_cells, draw_numbers
from experiments.global_params import cred, running_in, cgreen, cblue
from experiments.tools import make_image_high_resolution

image_format = '.png'  # .pdf
num_cells_per_dim = [30]

fv_presentation_path = Path.joinpath(config.subcell_presentation_path, 'finite_volumes')
fv_presentation_path.mkdir(parents=True, exist_ok=True)

names_dict = {
    "aero_qelvira_vertex": "ELVIRA-WO + AEROS Quadratic + TEM + AEROS Vertex",
    "elvira_w_oriented": "ELVIRA",
    "quadratic_aero": "AEROS Quadratic",
    "upwind": "UpWind",
}
chosen_models = ["upwind", "elvira_w_oriented", "quadratic_aero", "aero_qelvira_vertex"]
model_color = {
    "upwind": "#FFFF00",
    "elvira_w_oriented": "#FC6255",
    "quadratic_aero": "#58C4DD",
    "aero_qelvira_vertex": "#CAA3E8",
}
names_dict = {k: names_dict[k] for k in model_color.keys()}


@perplex_plot(legend=False)
@one_line_iterator()
def plot_upwind_reconstruction_init_final(fig, ax, true_reconstruction, num_cells_per_dim, resolution,
                                          reconstruction, alpha_true_image=0.5, numbers_on=False,
                                          cmap_true_image="Greys_r",
                                          trim=((0, 1), (0, 1)), vmin=None, vmax=None, labels=True):
    model_resolution = np.array(resolution)
    image = true_reconstruction[0]

    if alpha_true_image > 0:
        plot_cells(ax, colors=image, mesh_shape=model_resolution, alpha=alpha_true_image, cmap=cmap_true_image,
                   vmin=np.min(image) if vmin is None else vmin,
                   vmax=np.max(image) if vmax is None else vmax,
                   labels=labels)

    plot_cells(ax, colors=make_image_high_resolution(reconstruction[-1],
                                                     np.array(np.shape(image)) / np.array(
                                                         np.shape(reconstruction[-1]))),
               mesh_shape=model_resolution,
               alpha=0.5, cmap=cmap_true_image,
               vmin=0,
               vmax=1,
               labels=labels)

    ax.set_ylim((model_resolution[1] - trim[0][1] - 0.5, -0.5 + trim[0][0]))
    ax.set_xlim((trim[1][0] - 0.5, model_resolution[0] - trim[1][1] - 0.5))

    draw_numbers(
        ax, mesh_shape=num_cells_per_dim,
        refinement=model_resolution // num_cells_per_dim,
        numbers_on=numbers_on,
        prop_ticks=10 / num_cells_per_dim  # each 10 cells a tick
    )

    if not numbers_on:
        plt.box(False)


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.paper_results_path,
        emissions_path=config.results_path,
        name=f'SchemesRev',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )
    data_manager.load()

    # ---------------- Paper plots ---------------- #
    kwargs4savefig = {"bbox_inches": 'tight', "pad_inches": 0, "transparent": True}
    for ncpd, legend in zip(num_cells_per_dim, [True] * len(num_cells_per_dim)):
        for i in range(len(chosen_models)):
            generic_plot(data_manager,
                         name=f"ReconstructionErrorInTimex{i}",
                         format=image_format,
                         path=fv_presentation_path,
                         x="times", y="scheme_error", label="method",
                         # axes_by=["num_cells_per_dim"],
                         num_cells_per_dim=ncpd,
                         add_legend=legend,
                         plot_by=["num_cells_per_dim", "image", "angular_velocity", "velocity"],
                         times=lambda ntimes, num_screenshots: np.linspace(0, ntimes, num_screenshots, dtype=int),
                         scheme_error=scheme_reconstruction_error,
                         plot_func=NamedPartial(
                             sns.lineplot,
                             marker="o", linestyle="--",
                             markers=True, linewidth=3,
                             palette={v: model_color[k] for k, v in names_dict.items()}
                         ),
                         axes_xy_proportions=(12, (5 if legend else 4.2)), transpose=True,
                         share_axis="x", log="y",
                         models=chosen_models[:(i + 1)],
                         method=lambda models: names_dict[models],
                         # num_cells_per_dim=[30],
                         axis_font_dict={'color': 'black', 'weight': 'normal', 'size': 25},
                         labels_font_dict={'color': 'black', 'weight': 'normal', 'size': 25},
                         legend_font_dict={'weight': 'normal', "size": 19, 'stretch': 'normal'},
                         uselatex=False if running_in == "server" else True,
                         xlabel=r"Iterations",
                         ylabel=r"$\|u-\tilde u \|_{L^1}$",
                         # xticks=[1, 2, 4, 8, 16, 24, 40, 70, 120],
                         create_preimage_data=True if running_in == "server" else False,
                         use_preimage_data=True,
                         only_create_preimage_data=True if running_in == "server" else False,
                         kwargs4savefig=kwargs4savefig,
                         legend_outside_plot=LegendOutsidePlot(loc="lower center",
                                                               extra_y_top=0.01,
                                                               extra_y_bottom=0.15 + (0.25 if legend else 0),
                                                               extra_x_left=0.125, extra_x_right=0.075),
                         extra_plot_processes=extra_process4plot,
                         ylim=(0.3e-2, None),
                         )

    kwargs4savefig = {"bbox_inches": 'tight', "pad_inches": 0}
    for i, ncpd in enumerate(num_cells_per_dim):
        plot_reconstruction_init_final(
            data_manager,
            path=fv_presentation_path,
            format=image_format,
            plot_by=["num_cells_per_dim", "angular_velocity", "image", "velocity", "models"],
            num_cells_per_dim=ncpd,
            folder="InitEndComparison",
            alpha_true_image=0.1,
            cmap_true_image="Greys_r", init_end_color=("black", cred),
            image=[
                "batata.jpg",
                "ShapesVertex.jpg",
            ],
            kwargs4savefig=kwargs4savefig,
            trim=((2 * (i + 1), 3 * (i + 1)), (2 * (i + 1), 3 * (i + 1))), vmin=None, vmax=None, labels=False)
        plot_upwind_reconstruction_init_final(
            data_manager,
            path=fv_presentation_path,
            format=image_format,
            models=["upwind"],
            plot_by=["num_cells_per_dim", "angular_velocity", "image", "velocity", "models"],
            num_cells_per_dim=ncpd,
            folder="InitEndComparison",
            alpha_true_image=0.1,
            cmap_true_image="Greys_r", init_end_color=("black", cred),
            image=[
                "batata.jpg",
                "ShapesVertex.jpg",
            ],
            kwargs4savefig=kwargs4savefig,
            trim=((2 * (i + 1), 3 * (i + 1)), (2 * (i + 1), 3 * (i + 1))), vmin=None, vmax=None, labels=False)

    for i, ncpd in enumerate(num_cells_per_dim):
        plot_reconstruction_init_final(
            data_manager,
            path=fv_presentation_path,
            format=image_format,
            plot_by=["num_cells_per_dim", "angular_velocity", "image", "velocity", "models"],
            folder="InitEndComparison",
            alpha_true_image=0.1,
            cmap_true_image="Greys", init_end_color=("black", cred),
            image="zalesak_notched_circle.jpg",
            kwargs4savefig=kwargs4savefig,
            trim=((2 * (i + 1), 2 * (i + 1)), (2 * (i + 1), 2 * (i + 1))), vmin=None, vmax=None, labels=False)
        plot_upwind_reconstruction_init_final(
            data_manager,
            path=fv_presentation_path,
            format=image_format,
            models=["upwind"],
            plot_by=["num_cells_per_dim", "angular_velocity", "image", "velocity", "models"],
            num_cells_per_dim=ncpd,
            folder="InitEndComparison",
            alpha_true_image=0.1,
            cmap_true_image="Greys_r", init_end_color=("black", cred),
            image="zalesak_notched_circle.jpg",
            kwargs4savefig=kwargs4savefig,
            trim=((2 * (i + 1), 3 * (i + 1)), (2 * (i + 1), 3 * (i + 1))), vmin=None, vmax=None, labels=False)

    print("CO2 consumption: ", data_manager.CO2kg)
    copy_main_script_version(__file__, data_manager.path)
