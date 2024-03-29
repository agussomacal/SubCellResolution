import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial, copy_main_script_version
from PerplexityLab.visualization import generic_plot, make_data_frames, LegendOutsidePlot
from experiments.PaperPlots.paper_corners_plot import aero_qelvira_vertex
from experiments.PaperPlots.exploring_methods_convergence import quadratic_aero, elvira_w_oriented
from experiments.PaperPlots.exploring_scheme_methods import scheme_error, plot_reconstruction_time_i, scheme_reconstruction_error, \
    plot_time_i, calculate_true_solution, fit_model
from experiments.global_params import cpink, corange, cblue, cgreen, runsinfo, cpurple, cred, cgray, \
    RESOLUTION_FACTOR, num_cores, running_in, only_create_preimage_data, image_format
from experiments.PaperPlots.models2compare import upwind

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

    ntimes = 120
    # ---------------- Paper plots ---------------- #
    generic_plot(data_manager,
                 name="ReconstructionErrorInTimex",
                 format=image_format,
                 path=config.subcell_paper_figures_path,
                 x="times", y="scheme_error", label="method",
                 plot_by=["num_cells_per_dim", "image"],
                 times=lambda ntimes: np.arange(0, ntimes, SAVE_EACH) + 1,
                 scheme_error=scheme_reconstruction_error,
                 plot_func=NamedPartial(
                     sns.lineplot,
                     marker="o", linestyle="--",
                     markers=True, linewidth=3,
                     palette={v: model_color[k] for k, v in names_dict.items()}
                 ),
                 log="yx",
                 models=list(model_color.keys()),
                 method=lambda models: names_dict[models],
                 # num_cells_per_dim=[30],
                 axes_xy_proportions=(12, 8),
                 axis_font_dict={'color': 'black', 'weight': 'normal', 'size': 25},
                 labels_font_dict={'color': 'black', 'weight': 'normal', 'size': 25},
                 legend_font_dict={'weight': 'normal', "size": 20, 'stretch': 'normal'},
                 uselatex=False if running_in == "server" else True,
                 xlabel=r"Iterations",
                 ylabel=r"$\|u-\tilde u \|_{L^1}$",
                 xticks=[1, 2, 4, 8, 16, 24, 40, 70, 120],
                 create_preimage_data=True if running_in == "server" else False,
                 use_preimage_data=True,
                 only_create_preimage_data=True if running_in == "server" else False,
                 legend_outside_plot=LegendOutsidePlot(loc="lower center",
                                                       extra_y_top=0.01, extra_y_bottom=0.3,
                                                       extra_x_left=0.125, extra_x_right=0.075),
                 )

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
