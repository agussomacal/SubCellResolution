import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial, copy_main_script_version
from PerplexityLab.visualization import generic_plot, make_data_frames
from experiments.PaperPlots.paper_corners import aero_qelvira_vertex
from experiments.subcell_paper.ex_aero import quadratic_aero, elvira_w_oriented
from experiments.subcell_paper.ex_scheme import scheme_error, plot_reconstruction_time_i, scheme_reconstruction_error, \
    plot_time_i, calculate_true_solution, fit_model
from experiments.subcell_paper.global_params import cpink, corange, cblue, cgreen, runsinfo, cpurple, cred, cgray, \
    RESOLUTION_FACTOR, num_cores, running_in
from experiments.subcell_paper.models2compare import upwind

SAVE_EACH = 1

# ========== ========== Names and colors to present ========== ========== #
names_dict = {
    "aero_qelvira_vertex": "ELVIRA-WO + AEROS Quadratic + TEM + AEROS Vertex",

    "elvira": "ELVIRA",
    "elvira_w_oriented": "ELVIRA-WO",

    "linear_obera": "OBERA Linear",
    "linear_obera_w": "OBERA-W Linear",

    "quadratic_obera_non_adaptive": "OBERA Quadratic",
    "quadratic_aero": "AEROS Quadratic",

    "upwind": "UpWind",
}
model_color = {
    "aero_qelvira_vertex": cblue,

    "elvira": corange,
    "elvira_w_oriented": cred,

    "linear_obera": cpink,
    "linear_obera_w": cpurple,

    "quadratic_obera_non_adaptive": cgreen,
    "quadratic_aero": cgreen,

    "upwind": cgray,
}
names_dict = {k: names_dict[k] for k in model_color.keys()}

runsinfo.append_info(
    **{k.replace("_", "-"): v for k, v in names_dict.items()}
)

if __name__ == "__main__":
    data_manager = DataManager(
        path=config.paper_results_path,
        emissions_path=config.results_path,
        name='Schemes',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )

    lab = LabPipeline()

    lab.define_new_block_of_functions(
        "ground_truth",
        calculate_true_solution,
        recalculate=False
    )

    lab.define_new_block_of_functions(
        "models",
        *map(fit_model, [
            aero_qelvira_vertex,
            elvira_w_oriented,
            quadratic_aero,
            upwind,
        ]),
        recalculate=False
    )

    ntimes = 120 if running_in == "server" else 20
    lab.execute(
        data_manager,
        num_cores=num_cores,
        forget=False,
        save_on_iteration=None,
        refinement=[1],
        ntimes=[ntimes],
        velocity=[(0, 1 / 4)],
        num_cells_per_dim=[15, 30],  # 60
        noise=[0],
        image=[
            "batata.jpg",
            "ShapesVertex.jpg",
        ],
        reconstruction_factor=[RESOLUTION_FACTOR],
    )
    print(set(data_manager["models"]))

    num_ticks = 10
    for log, xticks in zip(["", "x"], [np.arange(0, ntimes, ntimes // num_ticks, dtype=int),
                                       # np.logspace(0, np.log10(ntimes), num_ticks, dtype=int),
                                       [1, 2, 4, 8, 16, 24, 40, 70, 120]
                                       ]):
        generic_plot(data_manager,
                     name="ReconstructionErrorInTime" + log,
                     format=".pdf",
                     # path=config.subcell_paper_figures_path,
                     x="times", y="scheme_error", label="method",
                     plot_by=["num_cells_per_dim", "image"],
                     times=lambda ntimes: np.arange(0, ntimes, SAVE_EACH) + (1 if log else 0),
                     scheme_error=scheme_reconstruction_error,
                     plot_func=NamedPartial(
                         sns.lineplot,
                         marker="o", linestyle="--",
                         markers=True, linewidth=3,
                         palette={v: model_color[k] for k, v in names_dict.items()}
                     ),
                     log="y" + log,
                     models=list(model_color.keys()),
                     method=lambda models: names_dict[models],
                     axes_xy_proportions=(12, 8),
                     axis_font_dict={'color': 'black', 'weight': 'normal', 'size': 25},
                     legend_font_dict={'weight': 'normal', "size": 20, 'stretch': 'normal'},
                     uselatex=False if running_in == "server" else True,
                     xlabel=r"Iterations",
                     ylabel=r"$||u-\tilde u ||_{L^1}$",
                     xticks=xticks,
                     )

        generic_plot(data_manager,
                     name="ErrorInTime" + log,
                     format=".pdf", ntimes=ntimes,
                     path=config.subcell_paper_figures_path,
                     x="times", y="scheme_error", label="method",
                     plot_by=["num_cells_per_dim", "image"],
                     times=lambda ntimes: np.arange(1, ntimes + 1),
                     scheme_error=scheme_error,
                     plot_func=NamedPartial(
                         sns.lineplot, marker="o", linestyle="--",
                         palette={v: model_color[k] for k, v in names_dict.items()}
                     ),
                     models=list(model_color.keys()),
                     method=lambda models: names_dict[models],
                     axis_font_dict={'color': 'black', 'weight': 'normal', 'size': 25},
                     legend_font_dict={'weight': 'normal', "size": 20, 'stretch': 'normal'},
                     xlabel=r"Iterations",
                     ylabel=r"$||a-\tilde a ||_{\ell^1}$",
                     log="y" + log,
                     xticks=xticks,
                     uselatex=False if running_in == "server" else True,
                     )

    for i in range(0, ntimes, SAVE_EACH):
        plot_reconstruction_time_i(
            data_manager,
            # path=config.subcell_paper_figures_path,
            # format=".pdf",
            i=i // SAVE_EACH,
            name=f"Reconstruction{i}",
            folder='Reconstruction',
            folder_by=['image', 'num_cells_per_dim'],
            models=["aero_qelvira_vertex"],
            plot_by=["method"],
            axes_by=[],
            # models=[model for model in model_color.keys() if "flux" not in model],
            method=lambda models: names_dict[models],
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
            uselatex=False if running_in == "server" else True,
        )

    for i in range(ntimes):
        plot_time_i(data_manager, folder="Solution", name=f"Time{i}", i=i, alpha=0.8, cmap="viridis",
                    trim=((0, 0), (0, 0)), folder_by=['image', 'num_cells_per_dim'],
                    plot_by=[],
                    axes_by=["method"],
                    models=list(model_color.keys()),
                    method=lambda models: names_dict[models],
                    numbers_on=True, error=True)
    # ========== =========== ========== =========== #
    #               Experiment Times                #
    # ========== =========== ========== =========== #
    # times to fit cell
    df = next(make_data_frames(
        data_manager,
        var_names=["models", "time_to_fit"],
        group_by=[],
    ))[1].groupby("models").mean()["time_to_fit"] / ntimes
    runsinfo.append_info(
        **{"scheme-" + k.replace("_", "-") + "-time": f"{v:.1g}" for k, v in df.items()}
    )

    print("CO2 consumption: ", data_manager.CO2kg)
    copy_main_script_version(__file__, data_manager.path)
