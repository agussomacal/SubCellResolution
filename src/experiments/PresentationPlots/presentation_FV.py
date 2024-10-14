import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.miscellaneous import NamedPartial, copy_main_script_version
from PerplexityLab.visualization import generic_plot, LegendOutsidePlot
from experiments.PaperPlots.exploring_scheme_methods import scheme_reconstruction_error
from experiments.PaperPlots.paper_scheme_plot_transp import plot_reconstruction_init_final, model_color, names_dict, \
    num_cells_per_dim
from experiments.global_params import cred, running_in

image_format = '.png'  # .pdf

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
    for ncpd, legend in zip([30, 60], [False, True]):
        generic_plot(data_manager,
                     name="ReconstructionErrorInTimex",
                     format=image_format,
                     path=config.subcell_presentation_path,
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
                     models=list(model_color.keys()),
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
                     legend_outside_plot=LegendOutsidePlot(loc="lower center",
                                                           extra_y_top=0.01,
                                                           extra_y_bottom=0.15 + (0.25 if legend else 0),
                                                           extra_x_left=0.125, extra_x_right=0.075),
                     )

    for i, ncpd in enumerate(num_cells_per_dim):
        plot_reconstruction_init_final(
            data_manager,
            path=config.subcell_presentation_path,
            format=".pdf",
            plot_by=["num_cells_per_dim", "angular_velocity", "image", "velocity", "models"],
            num_cells_per_dim=ncpd,
            folder="InitEndComparison",
            alpha_true_image=0.1,
            cmap_true_image="Greys_r", init_end_color=("black", cred),
            image=[
                "batata.jpg",
                "ShapesVertex.jpg",
            ],
            trim=((2 * (i + 1), 3 * (i + 1)), (2 * (i + 1), 3 * (i + 1))), vmin=None, vmax=None, labels=False)

    for i, ncpd in enumerate(num_cells_per_dim):
        plot_reconstruction_init_final(
            data_manager,
            path=config.subcell_presentation_path,
            format=".pdf",
            plot_by=["num_cells_per_dim", "angular_velocity", "image", "velocity", "models"],
            folder="InitEndComparison",
            alpha_true_image=0.1,
            cmap_true_image="Greys", init_end_color=("black", cred),
            image="zalesak_notched_circle.jpg",
            trim=((2 * (i + 1), 2 * (i + 1)), (2 * (i + 1), 2 * (i + 1))), vmin=None, vmax=None, labels=False)

    print("CO2 consumption: ", data_manager.CO2kg)
    copy_main_script_version(__file__, data_manager.path)
