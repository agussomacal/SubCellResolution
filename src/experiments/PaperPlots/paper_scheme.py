import time

import numpy as np
import seaborn as sns
from tqdm import tqdm

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import generic_plot
from experiments.PaperPlots.paper_corners import aero_qelvira_vertex
from experiments.subcell_paper.ex_aero import piecewise_constant, \
    quadratic_aero, quartic_aero, elvira, elvira_w_oriented, linear_obera, linear_obera_w, \
    quadratic_obera_non_adaptive, elvira_w
from experiments.subcell_paper.ex_scheme import scheme_error, plot_reconstruction_time_i, scheme_reconstruction_error, \
    plot_time_i, calculate_true_solution, fit_model
from experiments.subcell_paper.global_params import cpink, corange, cyellow, \
    cblue, cgreen, runsinfo, EVALUATIONS, cpurple, cred, ccyan, cgray
from experiments.subcell_paper.models2compare import upwind
from experiments.subcell_paper.tools import calculate_averages_from_image, load_image, \
    reconstruct, singular_cells_mask
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.LearningFluxRegularCellCreator import CellLearnedFlux
from lib.SubCellScheme import SubCellScheme

SAVE_EACH = 1

# ========== ========== Names and colors to present ========== ========== #
names_dict = {
    "aero_qelvira_vertex": "ELVIRA + TEM + AEROS Quadratic + AVROS",
    # "aero_qelvira_tem": "ELVIRA + TEM + AEROS Quadratic",
    # "aero_qvertex": "AEROS Quadratic + TEM + AVROS",
    # "aero_qtem": "AEROS Quadratic + TEM",
    # "obera_qtem": "OBERA Quadratic + TEM",

    "elvira": "ELVIRA",
    "elvira_w_oriented": "ELVIRA W-Oriented",

    "linear_obera": "LINEAR OBERA",
    "linear_obera_w": "LINEAR OBERA WO",

    "quadratic_obera_non_adaptive": "OBERA QUADRATIC",
    "quadratic_aero": "AEROS Quadratic",

    "upwind": "UpWind",
}
model_color = {
    "aero_qelvira_vertex": cblue,
    # "aero_qelvira_tem": "ELVIRA + TEM + AEROS Quadratic",
    # "aero_qvertex": "AEROS Quadratic + TEM + AVROS",
    # "aero_qtem": "AEROS Quadratic + TEM",
    # "obera_qtem": "OBERA Quadratic + TEM",

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
            # aero_qelvira_vertex,
            # aero_qelvira_tem,
            # aero_qvertex,
            # aero_qtem,

            # piecewise_constant,
            # elvira,
            elvira_w,
            # elvira_w_oriented,

            # linear_obera,
            # linear_obera_w,

            # quadratic_obera_non_adaptive,
            quadratic_aero,

            # quartic_aero,
            upwind,
        ]),
        recalculate=False
    )

    ntimes = 20
    lab.execute(
        data_manager,
        num_cores=15,
        forget=False,
        save_on_iteration=None,
        refinement=[1],
        ntimes=[ntimes],
        velocity=[(0, 1 / 4)],
        num_cells_per_dim=[30],  # 60
        noise=[0],
        image=[
            "batata.jpg"
            # "ShapesVertex_1680x1680.jpg",
        ],
        reconstruction_factor=[5],
        # reconstruction_factor=[1],
    )
    print(set(data_manager["models"]))

    generic_plot(data_manager,
                 name="ErrorInTime",
                 format=".pdf", ntimes=ntimes,
                 # path=config.subcell_paper_figures_path,
                 x="times", y="scheme_error", label="method", plot_by=["num_cells_per_dim", "image"],
                 # models=["elvira", "quadratic"],
                 times=lambda ntimes: np.arange(1, ntimes + 1),
                 scheme_error=scheme_error,
                 plot_func=NamedPartial(
                     sns.lineplot, marker="o", linestyle="--",
                     palette={v: model_color[k] for k, v in names_dict.items()}
                 ),
                 models=list(model_color.keys()),
                 method=lambda models: names_dict[models],
                 log="y",
                 )
    #
    # new_times = np.array([1, 7, 13, 19, 26, 32, 38, 44, 51, 57, 63, 69, 76, 82, 88, 94, 101, 107, 113, 119])
    generic_plot(data_manager,
                 name="ReconstructionErrorInTime",
                 format=".pdf",
                 # path=config.subcell_paper_figures_path,
                 x="times", y="scheme_error", label="method", plot_by=["num_cells_per_dim", "image"],
                 # models=["elvira", "quadratic"],
                 times=lambda ntimes: np.arange(0, ntimes, SAVE_EACH),
                 # times=lambda ntimes: new_times,
                 scheme_error=scheme_reconstruction_error,
                 plot_func=NamedPartial(
                     sns.lineplot, marker="o", linestyle="--",
                     palette={v: model_color[k] for k, v in names_dict.items()}
                 ),
                 log="y",
                 models=list(model_color.keys()),
                 method=lambda models: names_dict[models],
                 )

    generic_plot(data_manager,
                 name="Time2Fit",
                 x="image", y="time_to_fit", label="method", plot_by=["num_cells_per_dim", "image"],
                 # models=["elvira", "quadratic"],
                 times=lambda ntimes: np.arange(ntimes),
                 plot_func=NamedPartial(
                     sns.barplot,
                     palette={v: model_color[k] for k, v in names_dict.items()}
                 ),
                 log="y",
                 models=list(model_color.keys()),
                 method=lambda models: names_dict[models],
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
            plot_by=[],
            axes_by=["method"],
            models=[model for model in model_color.keys() if "flux" not in model],
            method=lambda models: names_dict[models],
            # plot_by=['image', 'models', 'num_cells_per_dim'],
            # folder_by=["image"],
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
            # trim=((3, 3), (3, 3)),
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
            # winner_color_dict=winner_color_dict
        )

    for i in range(ntimes):
        plot_time_i(data_manager, folder="Solution", name=f"Time{i}", i=i, alpha=0.8, cmap="viridis",
                    trim=((0, 0), (0, 0)), folder_by=['image', 'num_cells_per_dim'],
                    plot_by=[],
                    axes_by=["method"],
                    models=list(model_color.keys()),
                    method=lambda models: names_dict[models],
                    numbers_on=True, error=True)
