from collections import OrderedDict
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

import config
from PerplexityLab.DataManager import DataManager, JOBLIB, dmfilter
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial, copy_main_script_version
from PerplexityLab.visualization import generic_plot, make_data_frames, save_fig
from experiments.MLTraining.ml_global_params import num_cores
from experiments.VizReconstructionUtils import plot_image
from experiments.subcell_paper.ex_aero import obtain_images, obtain_image4error, fit_model, piecewise_constant, \
    quadratic_aero, quartic_aero, PlotStyle, elvira, elvira_w_oriented, linear_obera, linear_obera_w, \
    quadratic_obera_non_adaptive, plot_convergence, plot_reconstruction
from experiments.subcell_paper.global_params import SUB_CELL_DISCRETIZATION2BOUND_ERROR, runsinfo, cblue, cgreen, cred, \
    cpurple, cpink
from experiments.subcell_paper.tools import curve_cells_fitting_times

num_cells_per_dim_2plot = [10, 15, 30] + [14, 21, 28]
num_cells_per_dim = list(map(int, np.unique(np.logspace(np.log10(20), np.log10(100), num=20, dtype=int).tolist() +
                                            np.logspace(np.log10(10), np.log10(20), num=5, dtype=int,
                                                        endpoint=False).tolist() +
                                            num_cells_per_dim_2plot)))

accepted_models = {
    "HighOrderModels": OrderedDict([
        ("piecewise_constant", PlotStyle(color=cpink, marker="o", linestyle="--")),

        ("linear_obera", PlotStyle(color=cblue, marker=".", linestyle=":")),
        ("elvira", PlotStyle(color=cblue, marker="o", linestyle="--")),
        ("elvira_w_oriented", PlotStyle(color=cpurple, marker="o", linestyle="--")),
        ("linear_obera_w", PlotStyle(color=cpurple, marker=".", linestyle=":")),

        ("quadratic_aero", PlotStyle(color=cgreen, marker="o", linestyle="--")),
        ("quadratic_obera_non_adaptive", PlotStyle(color=cgreen, marker=".", linestyle=":")),

        ("quartic_aero", PlotStyle(color=cred, marker="o", linestyle="--")),
        # ("quartic_obera", PlotStyle(color=cpurple, marker=".", linestyle=":")),
    ]),
}

names_dict = {
    "piecewise_constant": "Piecewise Constant",

    "elvira": "ELVIRA",
    # "elvira_w": "ELVIRA-W",
    "elvira_w_oriented": "ELVIRA-WO",

    "linear_obera": "OBERA Linear",  # l1
    "linear_obera_w": "OBERA-W Linear",  # l1

    "quadratic_obera_non_adaptive": "OBERA Quadratic",
    "quadratic_aero": "AEROS Quadratic",

    "cubic_aero": "AEROS Cubic",
    "cubic_obera": "OBERA Cubic",

    "quartic_aero": "AEROS Quartic",
    "quartic_obera": "OBERA Quartic",
}

rateonly = list(filter(lambda x: "circle" not in x, names_dict.keys()))
runsinfo.append_info(
    **{k.replace("_", "-"): v for k, v in names_dict.items()}
)

# ========== =========== ========== =========== #
#                Experiment Run                 #
# ========== =========== ========== =========== #
data_manager = DataManager(
    path=config.paper_results_path,
    emissions_path=config.results_path,
    name=f'PaperAERO',
    format=JOBLIB,
    trackCO2=True,
    country_alpha_code="FR"
)
# data_manager.load()

lab = LabPipeline()
lab.define_new_block_of_functions(
    "precompute_images",
    obtain_images
)

lab.define_new_block_of_functions(
    "precompute_error_resolution",
    obtain_image4error
)

lab.define_new_block_of_functions(
    "models",
    *list(map(fit_model,
              [
                  piecewise_constant,
                  elvira,
                  elvira_w_oriented,

                  linear_obera,
                  linear_obera_w,

                  quadratic_obera_non_adaptive,
                  quadratic_aero,

                  quartic_aero,
                  # quartic_obera,
              ]
              )),
    recalculate=False
)

lab.execute(
    data_manager,
    num_cores=num_cores,
    forget=False,
    save_on_iteration=100,
    num_cells_per_dim=num_cells_per_dim,
    noise=[0],
    shape_name=[
        "Circle"
    ],
    sub_discretization2bound_error=[SUB_CELL_DISCRETIZATION2BOUND_ERROR],
)
# ========== =========== ========== =========== #
#               Experiment Plots                #
# ========== =========== ========== =========== #
circle_image = dmfilter(data_manager, names=["image4error"],
                        num_cells_per_dim=[max(num_cells_per_dim)])["image4error"][0]
with save_fig(paths=config.subcell_paper_figures_path, filename="Circle.pdf", show=False, dpi=None):
    plot_image(circle_image, cmap="viridis", vmin=-1, vmax=1, alpha=1)

circle_avg10 = dmfilter(data_manager, names=["image"], num_cells_per_dim=[10])["image"][0]
with save_fig(paths=config.subcell_paper_figures_path, filename="CircleAvg10.pdf", show=False, dpi=None):
    plot_image(circle_avg10, cmap="viridis", vmin=-1, vmax=1, alpha=1)

circle_avg30 = dmfilter(data_manager, names=["image"], num_cells_per_dim=[30])["image"][0]
with save_fig(paths=config.subcell_paper_figures_path, filename="CircleAvg30.pdf", show=False, dpi=None):
    plot_image(circle_avg30, cmap="viridis", vmin=-1, vmax=1, alpha=1)

# ----------- Color for model ---------- #
for group, model_style in accepted_models.items():
    runsinfo.append_info(
        **{f'color-{k.replace("_", "-")}': str(np.round(v.color, decimals=2).tolist())[1:-1] for k, v in
           model_style.items()}
    )

# ----------- Convergence ---------- #
for group, model_style in accepted_models.items():
    models2plot = list(model_style.keys())
    palette = {names_dict[k]: v.color for k, v in model_style.items()}

    generic_plot(data_manager,
                 name=f"Convergence_{group}",
                 path=config.subcell_paper_figures_path,
                 folder=group,
                 x="N", y="error_l1", label="models",
                 # num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(plot_convergence, model_style=model_style, names_dict=names_dict),
                 log="xy",
                 N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 models=models2plot,
                 method=lambda models: names_dict[str(models)],
                 names_dict=names_dict,
                 sorted_models=lambda models: models2plot.index(models),
                 sort_by=['sorted_models'],
                 format=".pdf",
                 axes_xy_proportions=(12, 8),
                 )

# ----------- Reconstruction ---------- #
for group, model_style in accepted_models.items():
    models2plot = list(model_style.keys())
    plot_reconstruction(
        data_manager,
        path=config.subcell_paper_figures_path,
        folder=group,
        format=".pdf",
        name=f"{group}",
        axes_by=['models'],
        models=models2plot,
        plot_by=['num_cells_per_dim', "models"],
        num_cells_per_dim=num_cells_per_dim_2plot,
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
        trim=((2 / 10, 5 / 10), (2 / 10, 5 / 10)),
        cmap="viridis",
        vmin=-1, vmax=1
    )


# ========== =========== ========== =========== #
#               Experiment Times                #
# ========== =========== ========== =========== #
def myround(n):
    # https://stackoverflow.com/questions/32812255/round-floats-down-in-python-to-keep-one-non-zero-decimal-only
    if n == 0:
        return 0
    sgn = -1 if n < 0 else 1
    scale = int(-np.floor(np.log10(abs(n))))
    if scale <= 0:
        scale = 1
    factor = 10 ** scale
    return sgn * np.floor(abs(n) * factor) / factor


# times to fit cell
df = next(make_data_frames(
    data_manager,
    var_names=["models", "time"],
    group_by=[],
    # models=models2plot,
    time=curve_cells_fitting_times,
))[1].groupby("models").apply(lambda x: np.nanmean(list(chain(*x["time"].values.tolist()))))
runsinfo.append_info(
    **{k.replace("_", "-") + "-time": np.round(v, decimals=4) for k, v in df.items()}
)

# times to fit cell std
dfstd = next(make_data_frames(
    data_manager,
    var_names=["models", "time"],
    group_by=[],
    # models=models2plot,
    time=curve_cells_fitting_times,
))[1].groupby("models").apply(lambda x: np.nanstd(list(chain(*x["time"].values.tolist()))))
runsinfo.append_info(
    **{"std-" + k.replace("_", "-") + "-time": np.round(v, decimals=4) for k, v in dfstd.items()}
)

dfstd = next(make_data_frames(
    data_manager,
    var_names=["models", "time"],
    group_by=[],
    # models=models2plot,
    time=curve_cells_fitting_times,
))[1].groupby("models").apply(lambda x: np.nanquantile(list(chain(*x["time"].values.tolist())), 0.05))
runsinfo.append_info(
    **{"qlow-" + k.replace("_", "-") + "-time": np.round(v, decimals=4) for k, v in dfstd.items()}
)

dfstd = next(make_data_frames(
    data_manager,
    var_names=["models", "time"],
    group_by=[],
    # models=models2plot,
    time=curve_cells_fitting_times,
))[1].groupby("models").apply(lambda x: np.nanquantile(list(chain(*x["time"].values.tolist())), 0.95))
runsinfo.append_info(
    **{"qhigh-" + k.replace("_", "-") + "-time": np.round(v, decimals=4) for k, v in dfstd.items()}
)

dfstd = next(make_data_frames(
    data_manager,
    var_names=["models", "time"],
    group_by=[],
    # models=models2plot,
    time=curve_cells_fitting_times,
))[1].groupby("models").apply(lambda x: np.nanquantile(list(chain(*x["time"].values.tolist())), 0.5))
runsinfo.append_info(
    **{"median-" + k.replace("_", "-") + "-time": np.round(v, decimals=4) for k, v in dfstd.items()}
)

print("CO2 consumption: ", data_manager.CO2kg)
copy_main_script_version(__file__, data_manager.path)
