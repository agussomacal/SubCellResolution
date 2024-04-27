from collections import OrderedDict
from itertools import chain

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import config
from PerplexityLab.DataManager import DataManager, JOBLIB, dmfilter
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial, copy_main_script_version
from PerplexityLab.visualization import generic_plot, make_data_frames, save_fig
from experiments.PaperPlots.exploring_methods_convergence import obtain_images, obtain_image4error, fit_model, \
    piecewise_constant, quadratic_aero, quartic_aero, PlotStyle, elvira, elvira_w_oriented, linear_obera, \
    linear_obera_w, quadratic_obera_non_adaptive, plot_reconstruction
from experiments.VizReconstructionUtils import plot_image
from experiments.global_params import SUB_CELL_DISCRETIZATION2BOUND_ERROR, runsinfo, cblue, cgreen, cred, \
    cpurple, cpink, running_in, only_create_preimage_data, image_format
from experiments.tools import curve_cells_fitting_times, calculate_averages_from_image, make_image_high_resolution, \
    load_image
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import ArrayIndexerNd, EXTEND
from lib.CellCreators.RegularCellCreator import PolynomialRegularCellCreator
from lib.CellIterators import iterate_by_condition_on_smoothness, iterate_all
from lib.CellOrientators import BaseOrientator
from lib.SmoothnessCalculators import indifferent
from lib.StencilCreators import StencilCreatorFixedShape
from lib.SubCellReconstruction import SubCellReconstruction, CellCreatorPipeline

image_format = ".png"


def save_fig_without_white(filename):
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()


def plot_convergence(data, x, y, hue, ax, threshold=30, rateonly=None, model_style=None, names_dict=None,
                     vlines=None, *args, **kwargs):
    for method, df in data.groupby(hue, sort=False):
        name = f"{names_dict[str(method)]}"
        if rateonly is None or method in rateonly:
            hinv = df[x].values
            valid_ix = hinv > threshold
            rate, origin = np.ravel(np.linalg.lstsq(
                np.vstack([np.log(hinv[valid_ix]), np.ones(np.sum(valid_ix))]).T,
                np.log(df[y].values[valid_ix]).reshape((-1, 1)), rcond=None)[0])
            name = fr"{name}: $\cal{{O}}$({abs(rate):.1f})"
        sns.lineplot(
            x=df[x], y=df[y], label=name, ax=ax, alpha=1,
            color=model_style[method].color if model_style is not None else None,
            marker=model_style[method].marker if model_style is not None else None,
            linestyle=model_style[method].linestyle if model_style is not None else None,
        )
    if vlines is not None:
        ax.vlines(vlines, linestyles=(0, (1, 8)), ymin=np.min(data[y]), ymax=np.max(data[y]),
                  colors='k', alpha=0.5)


if __name__ == "__main__":
    num_cells_per_dim_2plot_1 = [10, 15, 20, 25, 30]
    num_cells_per_dim_2plot_2 = [14, 21, 28]
    num_cells_per_dim_2plot_3 = [12, 24]
    num_cells_per_dim = list(map(int, np.unique(
        np.logspace(np.log10(20), np.log10(100), num=40 if running_in == "server" else 20, dtype=int).tolist() +
        np.logspace(np.log10(10), np.log10(20), num=5, dtype=int,
                    endpoint=False).tolist() +
        num_cells_per_dim_2plot_1 + num_cells_per_dim_2plot_2 + num_cells_per_dim_2plot_3)))

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
    data_manager.load()

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
                  ]
                  )),
        recalculate=False
    )

    # ========== =========== ========== =========== #
    #               Experiment Plots                #
    # ========== =========== ========== =========== #
    # cmap = "viridis"
    alpha = 1
    cmap = "coolwarm"
    vmin = 0
    vmax = 1

    # ---------- plot VOF ----------- #
    circle_image = dmfilter(data_manager, names=["image4error"],
                            num_cells_per_dim=[max(num_cells_per_dim)])["image4error"][0]
    # np.shape(circle_image) == 2000
    # 2000 == 2**4*5**3
    dx = 100
    dy = 50
    for i in range(np.shape(circle_image)[0] // 25):
        new_image = np.roll(circle_image, axis=(0, 1), shift=(i * dx, i * dy))
        plot_image(new_image, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
        save_fig_without_white(f"{config.subcell_presentation_path}/Circle_{i}{image_format}")
        for N in [10, 20]:
            avg_values = calculate_averages_from_image(new_image, num_cells_per_dim=N)
            plot_image(make_image_high_resolution(avg_values, reconstruction_factor=N), cmap=cmap, vmin=vmin, vmax=vmax,
                       alpha=alpha)
            save_fig_without_white(f"{config.subcell_presentation_path}/CircleAvg{N}_{i}{image_format}")

    # ---------- plot polynom reconstruction ----------- #
    subcell = SubCellReconstruction(
        name="Polynomials",
        smoothness_calculator=indifferent,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=iterate_all,  # only regular cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(3, 3)),
                cell_creator=PolynomialRegularCellCreator(degree=2, dimensionality=2)
            ),
        ],
    )
    i = 25
    image = np.load(f"{config.results_path}/Presentation/solution_avg_{i}.npy")[::-1]
    image -= np.min(image)
    image /= np.max(image)
    subcell.fit(image, indexer=ArrayIndexerNd(image, modes=EXTEND))
    reconstruction = subcell.reconstruct_by_factor(5)

    plot_image(reconstruction, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
    save_fig_without_white(f"{config.subcell_presentation_path}/solution_avg_{i}_reconstructed{image_format}")

    exit()
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

        vlines = [10, 20]
        generic_plot(data_manager,
                     name=f"Convergence_{group}_{'_'.join(map(str, vlines))}",
                     path=config.subcell_paper_figures_path,
                     folder=group,
                     x="num_cells_per_dim", y="error_l1", label="models",
                     plot_func=NamedPartial(plot_convergence, model_style=model_style, names_dict=names_dict,
                                            vlines=vlines,
                                            threshold=30),
                     log="xy",
                     models=models2plot,
                     method=lambda models: names_dict[str(models)],
                     sorted_models=lambda models: models2plot.index(models),
                     sort_by=['sorted_models'],
                     format=image_format,
                     axes_xy_proportions=(12, 8),
                     axis_font_dict={'color': 'black', 'weight': 'normal', 'size': 25},
                     labels_font_dict={'color': 'black', 'weight': 'normal', 'size': 25},
                     legend_font_dict={'weight': 'normal', "size": 19, 'stretch': 'normal'},
                     font_family="amssymb",
                     uselatex=False if running_in == "server" else True,
                     xlabel=r"$1/h$",
                     ylabel=r"$\|u-\tilde u \|_{L^1}$",
                     xticks=[10, 30, 100] + vlines,
                     create_preimage_data=True,
                     only_create_preimage_data=only_create_preimage_data
                     )

    # ----------- Reconstruction ---------- #
    for group, model_style in accepted_models.items():
        models2plot = list(model_style.keys())

        # for limits, ncpdg in zip([(2, 5), (3, 6), (3, 6)], [num_cells_per_dim_2plot_1, num_cells_per_dim_2plot_2, num_cells_per_dim_2plot_3]):
        # for limits, ncpdg in zip([(2, 5), (3, 6), (3, 6)],
        #                          [num_cells_per_dim_2plot_1, num_cells_per_dim_2plot_2, num_cells_per_dim_2plot_3]):
        #     for ncpd in ncpdg:
        limits = [2, 5]
        ncpdg = [10, 20]
        for ncpd in ncpdg:
            plot_reconstruction(
                data_manager,
                path=config.subcell_paper_figures_path,
                folder=group,
                format=image_format,
                name=f"{group}",
                models=models2plot,
                plot_by=['num_cells_per_dim', "models"],
                num_cells_per_dim=ncpd,
                axes_xy_proportions=(15, 15),
                difference=False,
                plot_curve=True,
                plot_curve_winner=False,
                plot_vh_classification=False,
                plot_singular_cells=False,
                alpha_true_image=1,
                alpha=0.65,
                numbers_on=False,
                plot_again=True,
                num_cores=1,
                trim=((limits[0] * ncpd / ncpdg[0], limits[1] * ncpd / ncpdg[0]),
                      (limits[0] * ncpd / ncpdg[0], limits[1] * ncpd / ncpdg[0])),
                cmap=cmap,
                cmap_true_image="Greys_r",
                vmin=vmin, vmax=vmax,
                labels=False,
                uselatex=False if running_in == "server" else True,
                create_preimage_data=True,
                only_create_preimage_data=only_create_preimage_data
            )

    # ========== =========== ========== =========== #
    #               Experiment Times                #
    # ========== =========== ========== =========== #
    # times to fit cell
    df = next(make_data_frames(
        data_manager,
        var_names=["models", "time"],
        group_by=[],
        # models=models2plot,
        time=curve_cells_fitting_times,
    ))[1].groupby("models").apply(lambda x: np.nanmean(list(chain(*x["time"].values.tolist()))))
    runsinfo.append_info(
        **{k.replace("_", "-") + "-time": f"{v:.1g}" for k, v in df.items()}
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
        **{"std-" + k.replace("_", "-") + "-time": f"{v:.1g}" for k, v in dfstd.items()}
    )

    dfstd = next(make_data_frames(
        data_manager,
        var_names=["models", "time"],
        group_by=[],
        # models=models2plot,
        time=curve_cells_fitting_times,
    ))[1].groupby("models").apply(lambda x: np.nanquantile(list(chain(*x["time"].values.tolist())), 0.05))
    runsinfo.append_info(
        **{"qlow-" + k.replace("_", "-") + "-time": f"{v:.1g}" for k, v in dfstd.items()}
    )

    dfstd = next(make_data_frames(
        data_manager,
        var_names=["models", "time"],
        group_by=[],
        # models=models2plot,
        time=curve_cells_fitting_times,
    ))[1].groupby("models").apply(lambda x: np.nanquantile(list(chain(*x["time"].values.tolist())), 0.95))
    runsinfo.append_info(
        **{"qhigh-" + k.replace("_", "-") + "-time": f"{v:.1g}" for k, v in dfstd.items()}
    )

    dfstd = next(make_data_frames(
        data_manager,
        var_names=["models", "time"],
        group_by=[],
        # models=models2plot,
        time=curve_cells_fitting_times,
    ))[1].groupby("models").apply(lambda x: np.nanquantile(list(chain(*x["time"].values.tolist())), 0.5))
    runsinfo.append_info(
        **{"median-" + k.replace("_", "-") + "-time": f"{v:.1g}" for k, v in dfstd.items()}
    )

    print("CO2 consumption: ", data_manager.CO2kg)
    copy_main_script_version(__file__, data_manager.path)
