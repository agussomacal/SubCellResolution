import warnings
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Generator

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from experiments.Refinement.ex_refinement_config import experiment_path, experiment_name, C_BLUE, C_PURPLE, C_RED, \
    C_GREEN, C_ORANGE, C_OLIVE, C_GRAY, C_BLACK
from experiments.Refinement.ex_refinement_convergence import get_label4plot, axis_font_dict, legend_font_dict, color
from experiments.Refinement.ex_refinement_models_to_compare import aero_3, aero_linear, quadratic, cubic
from experiments.Refinement.ex_refinement_singular_cells_connectivity import build_connected_singular_cell_graph
from experiments.Refinement.ex_refinement_tools import plx_fit_model, plx_obtain_image4error, fit_model
from experiments.global_params import cred
from experiments.tools import calculate_averages_from_curve
from lib.CellCreators.CellCreatorBase import REGULAR_CELL_TYPE
from lib.Curves.CurveCircle import CurveCircle, CircleParams
from lib.Curves.CurveTrigo import TrigoParams, CurveTrigo
from perplexitylab.experiment_tools import experiment_iterator, concatenate_iterators, define_default_constants, \
    define_default_variables, perplexifier, do
from perplexitylab.miscellaneous import filter_for_func
from perplexitylab.plot_tools import save_figure, sorted_legend

# 1680: 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 20, 21, 24, 28, 30, 35, 40, 42, 48, 56, 60, 70, 80, 84, 105, 120, 140, 168, 210, 240, 280, 336, 420, 560, 840, 1680
# divisors = [i for i in range(1, n+1) if n % i == 0]


file_format_data_to_plot = "csv"
filename_data_to_plot = "PairwiseSlopesPlot"
path_data_to_plot = f"{experiment_path}/{experiment_name}"

# Reconstruction plot params
matplotlib.rcParams['text.usetex'] = True
curve_color = cred
cmap_reconstruction = "Reds"
cmap_true_image = "Greys_r"
fig_size = (15, 15)


def get_versor(cell, slope):
    vec = [0, 0]
    vec[cell.independent_axis] = 1
    vec[cell.dependent_axis] = slope
    vec = np.array(vec)
    vec /= np.linalg.norm(vec)
    return vec


def get_ordered_slope_versors(model, graph):
    return [
        get_versor(
            cell=model.cells[c],
            slope=model.cells[c].curve.polynomial.deriv()(
                model.cells[c].center_of_cell[model.cells[c].independent_axis]))
        for c in graph
    ]


def get_pairwise_slope_versors(model, graph):
    # def get_slopes_versors(c, c_next):
    #     eval_coords = model.cells[c_next].center_of_cell - model.cells[c].center_of_cell
    #     slope = model.cells[c].curve.polynomial.deriv()(eval_coords[model.cells[c].independent_axis])
    #     slope_next = model.cells[c_next].curve.polynomial.deriv()(eval_coords[model.cells[c_next].independent_axis])
    #     return get_versor(model.cells[c], slope), get_versor(model.cells[c_next], slope_next)
    #
    # return [get_slopes_versors(graph[i], graph[i + 1]) for i in range(len(graph) - 1)]

    def get_slope_versor(c):
        slope = model.cells[c].curve.polynomial.deriv()(model.cells[c].center_of_cell[model.cells[c].independent_axis])
        return get_versor(model.cells[c], slope)

    return [(get_slope_versor(c=graph[i]), get_slope_versor(c=graph[i + 1])) for i in range(len(graph) - 1)]


def get_triplewise_slope_versors(model, graph):
    def get_slope_versor(c):
        slope = model.cells[c].curve.polynomial.deriv()(model.cells[c].center_of_cell[model.cells[c].independent_axis])
        return get_versor(model.cells[c], slope)

    return [(get_slope_versor(c=graph[i]), get_slope_versor(c=graph[i + 1]), get_slope_versor(c=graph[i + 1])) for i in
            range(len(graph) - 2)]


@perplexifier(default_path=experiment_path)
def single_experiment_continuity(shape, sub_cell_model, refinement, angle_threshold, num_cells_per_dim,
                                 trim=((2, 2), (-2, -2))):
    avg_values = calculate_averages_from_curve(shape, (num_cells_per_dim, num_cells_per_dim))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit_model(
            sub_cell_model=sub_cell_model,
            angle_threshold=angle_threshold,
            refinement=refinement, avg_values=avg_values)

    graph = build_connected_singular_cell_graph(
        singular_cells_coords=[cell.coords for cell in model.cells.values() if cell.CELL_TYPE != REGULAR_CELL_TYPE],
        model=model,
        trim=trim
    )

    return get_ordered_slope_versors(model, graph)


if __name__ == "__main__":
    # Experiment general params
    noise = 0
    seed = 42
    recalculate_all = False

    # ---------- Experiment list ---------- #
    iterator_builder, info = experiment_iterator(
        experiment_name=Path(__file__).stem,
        constants=define_default_constants(sub_cell_model=None, label=None, angle_threshold=0, reconstruction_factor=1),
        variables=define_default_variables(
            num_cells_per_dim=[40],
            shape=[
                CurveTrigo(params=TrigoParams(x0=0.5, y0=0.5, amplitude=0.1, frequency=1.)),
                CurveCircle(params=CircleParams(x0=0.511, y0=0.486, radius=0.232))
            ],
            # refinement=[1, 2, 3, 4, 5, 6, 7]
            refinement=[1, 2, 3, 4]
        ))


    def identifier(info):
        return f"Continuity_Img{info.shape}_{info.num_cells_per_dim}x{info.num_cells_per_dim}_{info.label}_Ref{info.refinement}"


    # ---------- Do experiments ---------- #
    experiment_labels_order = ["AEROS linear", "AEROS quadratic", "AEROS cubic"]

    iterators = []
    iterators.append(
        iterator_builder(sub_cell_model=aero_linear, label="AEROS linear",
                         angle_threshold=45, recalculate=False)
    )
    iterators.append(
        iterator_builder(sub_cell_model=quadratic, label="AEROS quadratic",
                         angle_threshold=45, recalculate=False)
    )
    iterators.append(
        iterator_builder(sub_cell_model=cubic, label="AEROS cubic",
                         angle_threshold=45, recalculate=False)
    )

    data = defaultdict(list)
    for experiment_info in concatenate_iterators(*iterators)():
        print("\n----------------------------------")
        print(identifier(experiment_info))
        ordered_versors = do(single_experiment_continuity, experiment_info)
        data["pairwise_versors"].append(list(zip(ordered_versors[:-1], ordered_versors[1:])))
        data["triplewise_versors"].append(list(zip(ordered_versors[:-2], ordered_versors[1:-1], ordered_versors[2:])))
        data["label"].append(experiment_info.label)
        data["refinement"].append(experiment_info.refinement)
        data["num_cells_per_dim"].append(experiment_info.num_cells_per_dim)
        data["shape"].append(str(experiment_info.shape))
    df = pd.DataFrame.from_dict(data)


    # ---------------------------------------------------------- #
    # Continuity convergence
    def diff_between_versors(pairwise_versors):
        return np.sqrt(np.sum(np.diff(pairwise_versors, axis=0) ** 2))


    def angle_between_versors(pairwise_versors, rad=False):
        a1 = np.arccos(np.dot(pairwise_versors[0], pairwise_versors[1]))
        a2 = np.arccos(np.dot(pairwise_versors[0], -pairwise_versors[1]))
        return np.min((a1, a2)) * (1 if rad else 180 / np.pi)

    def diff_angle_between_versors(triplewise_versors, rad=False, normalization=1,):
        a1 = angle_between_versors(triplewise_versors[:-1], rad=rad)
        a2 = angle_between_versors(triplewise_versors[1:], rad=rad)
        return (a2-a1)/normalization


    metrics = {
        "L1": lambda x: np.nanmean(np.abs(x)),
        "L2": lambda x: np.sqrt(np.nanmean(x ** 2)),
        "Loo": lambda x: np.nanmax(np.abs(x)),
    }
    metric_names = {
        "Loo": r"$L_\infty$",
        "L1": r"$L_1$",
        "L2": r"$L_2$",
    }

    linestyle = defaultdict(lambda: "solid")
    threshold_ref = 0

    for metric_name, metric in metrics.items():
        for (shape, num_cells_per_dim), sub_df in df.groupby(["shape", "num_cells_per_dim"]):
            labels_order = experiment_labels_order.copy()

            # ------------- FIRST ORDER differences plot ------------- #
            Y_VAR_NAME = "angle"
            with save_figure(filename=f"ContinuityConvergence_{shape}_{metric_name}", path=experiment_path,
                             figsize=(16, 8), show=False) as (fig, ax):
                sub_df = sub_df.groupby(["label", "refinement"]).apply(
                    lambda x: metric(
                        np.array(list(map(angle_between_versors, x["pairwise_versors"].values[0]))))).reset_index(
                    name=Y_VAR_NAME)
                for (label,), df2plot in sub_df.groupby(["label"]):
                    ref = df2plot["refinement"].copy()
                    valid_ix = ref >= threshold_ref
                    # rate, origin = np.ravel(np.linalg.lstsq(
                    #     np.vstack([np.log(ref[valid_ix]), np.ones(np.sum(valid_ix))]).T,
                    #     np.log(df2plot[Y_VAR_NAME].values[valid_ix]).reshape((-1, 1)), rcond=None)[0])
                    # label_plot_rate = fr"{label}: $1/h={num_cells_per_dim}$: $\cal{{O}}$({abs(rate):.1f})"

                    ref = df2plot["refinement"].copy()
                    rate, origin = np.ravel(np.linalg.lstsq(
                        np.vstack([ref[valid_ix], np.ones(np.sum(valid_ix))]).T,
                        np.log2(df2plot[Y_VAR_NAME].values[valid_ix]).reshape((-1, 1)), rcond=None)[0])

                    # plot points
                    label4plot = fr"{label}: $r={abs(rate):.1f}$"
                    labels_order[experiment_labels_order.index(label)] = label4plot  # replaces with the plot label
                    ax.plot(df2plot["refinement"], df2plot[Y_VAR_NAME], marker="o",
                            label=label4plot,
                            color=color[label], linestyle=linestyle[num_cells_per_dim])
                    # plot fitting line
                    # ax.plot(ref, 2 ** (origin + rate * ref),
                    #         color=C_BLACK, linestyle=linestyle[num_cells_per_dim], linewidth=1.5,
                    #         label=fr"Exponential fit: $r={abs(rate):.1f}$")

                # ax.set_xscale("log")
                ax.set_yscale("log")

                yticks = [6, 3, 1, 0.5, 0.1]
                # ax.set_xlim((int(min(xticks) * 0.9), int(max(xticks) * 1.1)))
                ax.set_yticks(yticks, labels=yticks)

                xticks = sorted(pd.unique(sub_df["refinement"]))
                ax.set_xticks(xticks, labels=list(map(str, np.array(xticks) - 1)))
                ax.grid(True)

                ax.set_title(fr"{shape} and metric {metric_names[metric_name]} $1/h={num_cells_per_dim}$")
                ax.set_xlabel(r"Number of subdivisions", fontdict=axis_font_dict)
                ax.set_ylabel(fr"{metric_names[metric_name]} angle (deg)", fontdict=axis_font_dict)
                sorted_legend(ax, labels_order, prop=legend_font_dict, loc='upper left', bbox_to_anchor=(1, 1))

                ax.tick_params(labelsize=axis_font_dict["size"])
                # ax.set_ylim((1e-7, 1e-1))
                ax.set_xlim((0, None))
                fig.tight_layout()

            # ------------- FIRST ORDER differences plot ------------- #
            Y_VAR_NAME = "angle difference"
            with save_figure(filename=f"Continuity2ndOrderConvergence_{shape}_{metric_name}", path=experiment_path,
                             figsize=(16, 8), show=False) as (fig, ax):
                sub_df = sub_df.groupby(["label", "refinement"]).apply(
                    lambda x: metric(
                        np.array(list(map(diff_angle_between_versors, x["triplewise_versors"].values[0]))))).reset_index(
                    name=Y_VAR_NAME)
                for (label,), df2plot in sub_df.groupby(["label"]):
                    ref = df2plot["refinement"].copy()
                    valid_ix = ref >= threshold_ref
                    # rate, origin = np.ravel(np.linalg.lstsq(
                    #     np.vstack([np.log(ref[valid_ix]), np.ones(np.sum(valid_ix))]).T,
                    #     np.log(df2plot[Y_VAR_NAME].values[valid_ix]).reshape((-1, 1)), rcond=None)[0])
                    # label_plot_rate = fr"{label}: $1/h={num_cells_per_dim}$: $\cal{{O}}$({abs(rate):.1f})"

                    ref = df2plot["refinement"].copy()
                    rate, origin = np.ravel(np.linalg.lstsq(
                        np.vstack([ref[valid_ix], np.ones(np.sum(valid_ix))]).T,
                        np.log2(df2plot[Y_VAR_NAME].values[valid_ix]).reshape((-1, 1)), rcond=None)[0])

                    # plot points
                    label4plot = fr"{label}: $r={abs(rate):.1f}$"
                    labels_order[experiment_labels_order.index(label)] = label4plot  # replaces with the plot label
                    ax.plot(df2plot["refinement"], df2plot[Y_VAR_NAME], marker="o",
                            label=label4plot,
                            color=color[label], linestyle=linestyle[num_cells_per_dim])
                    # plot fitting line
                    # ax.plot(ref, 2 ** (origin + rate * ref),
                    #         color=C_BLACK, linestyle=linestyle[num_cells_per_dim], linewidth=1.5,
                    #         label=fr"Exponential fit: $r={abs(rate):.1f}$")

                # ax.set_xscale("log")
                ax.set_yscale("log")

                yticks = [6, 3, 1, 0.5, 0.1]
                # ax.set_xlim((int(min(xticks) * 0.9), int(max(xticks) * 1.1)))
                ax.set_yticks(yticks, labels=yticks)

                xticks = sorted(pd.unique(sub_df["refinement"]))
                ax.set_xticks(xticks, labels=list(map(str, np.array(xticks) - 1)))
                ax.grid(True)

                ax.set_title(fr"{shape} and metric {metric_names[metric_name]} $1/h={num_cells_per_dim}$")
                ax.set_xlabel(r"Number of subdivisions", fontdict=axis_font_dict)
                ax.set_ylabel(fr"{metric_names[metric_name]} angle difference (deg)", fontdict=axis_font_dict)
                sorted_legend(ax, labels_order, prop=legend_font_dict, loc='upper left', bbox_to_anchor=(1, 1))

                ax.tick_params(labelsize=axis_font_dict["size"])
                # ax.set_ylim((1e-7, 1e-1))
                ax.set_xlim((0, None))
                fig.tight_layout()

    # exit()
    # ---------------------------------------------------------- #
    # Continuity histogram
    color = {
        1: C_GREEN,
        2: C_BLUE,
        3: C_PURPLE,
        4: C_RED,
        5: C_ORANGE,
        6: C_OLIVE,
    }

    for shape, sub_df in df.groupby("shape"):
        with save_figure(filename=f"HistogramContinuity_{shape}", path=experiment_path, figsize=(16, 8),
                         show=False) as (
                fig, ax):
            sub_df["label_plot"] = sub_df.apply(
                lambda x: get_label4plot(x["label"], x["refinement"]),
                axis=1)

            for (label_plot, label, refinement), df4plot in sub_df.groupby(["label_plot", "label", "refinement"]):
                angles = np.array(list(map(angle_between_versors, df4plot["pairwise_versors"].values[0])))
                angles = angles[angles > 1e-4]
                hist, bins = np.histogram(angles, bins=int(np.sqrt(len(angles))))
                logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
                ax.hist(angles, bins=logbins, color=color[refinement], label=label_plot, alpha=0.5, log=True)
                # plt.hist(angles, color=color[refinement], label=label_plot, alpha=0.5, log=True)
                ax.axvline(np.nanmedian(angles), color=color[refinement], linestyle="dashed", linewidth=2)

            ax.set_xscale("log")
            ax.set_yscale("log")

            xticks = [6, 3, 1, 0.5, 0.1]
            # ax.set_xlim((int(min(xticks) * 0.9), int(max(xticks) * 1.1)))
            ax.set_xticks(xticks, labels=xticks)
            ax.grid(True)

            ax.set_title(shape)
            ax.set_xlabel(r"Angle difference (deg)", fontdict=axis_font_dict)
            ax.set_ylabel("Counts", fontdict=axis_font_dict)
            ax.legend(prop=legend_font_dict, loc='upper left', bbox_to_anchor=(1, 1))
            ax.tick_params(labelsize=axis_font_dict["size"])
            # ax.set_ylim((1e-7, 1e-1))
            # ax.set_xlim((0, None))
            fig.tight_layout()
