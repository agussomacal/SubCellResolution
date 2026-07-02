import warnings
from collections import defaultdict
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from experiments.Refinement.ex_refinement_config import experiment_path, experiment_name, C_GREEN, C_BLUE, C_PURPLE, \
    C_ORANGE, C_RED
from experiments.Refinement.ex_refinement_models_to_compare import quadratic, aero_linear, elvira, elvira_w, cubic, \
    quadratic_lsq5
from experiments.Refinement.ex_refinement_tools import fit_model, calculate_error, \
    efficient_reconstruction, obtain_image4error
from experiments.tools import calculate_averages_from_curve
from lib.Curves.CurveCircle import CurveCircle, CircleParams
from lib.Curves.CurveTrigo import CurveTrigo, TrigoParams
from perplexitylab.experiment_tools import experiment_iterator, concatenate_iterators, define_default_constants, \
    define_default_variables, perplexifier
from perplexitylab.miscellaneous import filter_for_func
from perplexitylab.plot_tools import save_figure

file_format_data_to_plot = "csv"
filename_data_to_plot = "ConvergencePlot"
path_data_to_plot = f"{experiment_path}/{experiment_name}"

# Experiment general params
recalculate_all = False


def identifier(experiment_info):
    return f"Img{experiment_info.shape}_{experiment_info.num_cells_per_dim}x{experiment_info.num_cells_per_dim}_{experiment_info.label}_Ref{experiment_info.refinement}"


def get_label4plot(label, refinement):
    return f'{label}{" Subdivisions=" + str(refinement - 1) if refinement > 1 else ""}'


@perplexifier(default_path=experiment_path)
def single_experiment_convergence(shape, sub_cell_model, refinement, angle_threshold, num_cells_per_dim,
                                  sub_discretization2bound_error, p, evaluation_mode=False):
    avg_values = calculate_averages_from_curve(shape, (num_cells_per_dim, num_cells_per_dim))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit_model(
            sub_cell_model=sub_cell_model,
            angle_threshold=angle_threshold,
            refinement=refinement, avg_values=avg_values)
    true_reconstruction = obtain_image4error(
        shape=shape, num_cells_per_dim=num_cells_per_dim,
        sub_discretization2bound_error=sub_discretization2bound_error,
        avg_values=avg_values, evaluation_mode=evaluation_mode)
    reconstruction = efficient_reconstruction(
        model=model, avg_values=avg_values,
        sub_discretization2bound_error=sub_discretization2bound_error,
        refinement=refinement, evaluation_mode=evaluation_mode)
    error = calculate_error(true_reconstruction, reconstruction, p=p)
    # import matplotlib.pylab as plt
    # d = np.abs(true_reconstruction - reconstruction)
    # d = np.log10(d[d > 1e-10])
    # plt.hist(d)
    # plt.axvline(np.mean(d), c="k")
    # plt.show()
    return error


@perplexifier(default_path=experiment_path,
              filename=filename_data_to_plot,
              saver=lambda data, filepath: data.to_csv(filepath),
              loader=lambda filepath: pd.read_csv(filepath),
              file_format=file_format_data_to_plot)
def do_experiment_convergence(iterators: Tuple[Generator]):
    data = defaultdict(list)
    for experiment_info in concatenate_iterators(*iterators)():
        print("\n----------------------------------")
        print(identifier(experiment_info))
        _, error = single_experiment_convergence(
            recalculate=experiment_info.recalculate,
            **filter_for_func(single_experiment_convergence, experiment_info._asdict())
        )
        data["error"].append(error)
        data["label"].append(experiment_info.label)
        data["refinement"].append(experiment_info.refinement)
        data["num_cells_per_dim"].append(experiment_info.num_cells_per_dim)
        data["shape"].append(str(experiment_info.shape))
    return pd.DataFrame.from_dict(data)


if __name__ == "__main__":
    # ---------- Experiment variables and constants default ---------- #
    iterator_builder, info = experiment_iterator(
        experiment_name=Path(__file__).stem,
        constants=define_default_constants(sub_cell_model=None, label=None, angle_threshold=0, reconstruction_factor=1,
                                           sub_discretization2bound_error=2 * 2 * 3 * 2,
                                           # sub_discretization2bound_error=2 * 2 * 3,
                                           p=1, recalculate=False,
                                           recalculate_inner_funcs=False, evaluation_mode=False),
        variables=define_default_variables(
            # num_cells_per_dim=[20, 30, 40, 50, 60, 65],
            num_cells_per_dim=[20, 30, 40, 50, 60, 70, 80, 90, 100],
            shape=[
                CurveCircle(params=CircleParams(x0=0.511, y0=0.486, radius=0.232)),
                CurveTrigo(params=TrigoParams(x0=0.511, y0=0.486, amplitude=0.232, frequency=1.))
            ],
            refinement=[1, 2, ]
        ))

    # ---------- Do experiments ---------- #
    _, df = do_experiment_convergence(
        recalculate=True or recalculate_all,
        iterators=(
            # iterator_builder(sub_cell_model=cubic, label="AEROS cubic", refinement=[1, 2], angle_threshold=45,
            #                  recalculate=False or recalculate_all),
            iterator_builder(sub_cell_model=quadratic, label="AEROS quadratic", refinement=[1], angle_threshold=45,
                             recalculate=False or recalculate_all),
            iterator_builder(sub_cell_model=quadratic, label="AEROS quadratic", refinement=[2], angle_threshold=0,
                             recalculate=False or recalculate_all),
            # iterator_builder(sub_cell_model=quadratic_lsq5, label="AEROS quadratic lsqx5", refinement=[1, 2], angle_threshold=45,
            #                  recalculate=False or recalculate_all),
            iterator_builder(sub_cell_model=aero_linear, label="AEROS linear", refinement=[1, 2, 3], angle_threshold=45,
                             recalculate=False or recalculate_all),
            iterator_builder(sub_cell_model=elvira, label="ELVIRA", refinement=[1, 2, 3],
                             recalculate=False or recalculate_all),
            iterator_builder(sub_cell_model=elvira_w, label="ELVIRA W", refinement=[1, 2, 3],
                             num_cells_per_dim=[20, 30, 40, 50, 60, ],
                             recalculate=False or recalculate_all, recalculate_inner_funcs=False),
        ),
    )

    # ---------- Do plot ---------- #
    axis_font_dict = {'color': 'black', 'weight': 'normal', 'size': 25}
    labels_font_dict = {'color': 'black', 'weight': 'normal', 'size': 25}
    legend_font_dict = {'weight': 'normal', "size": 19, 'stretch': 'normal'}
    line_style = {1: "solid", 2: "dashed", 3: "dashdot", 4: "dotted"}
    marker_style = {1: "o", 2: "^", 3: "s", 4: ""}
    color = {
        "AEROS quadratic": C_GREEN,
        "AEROS linear": C_BLUE,
        "AEROS cubic": C_PURPLE,
        "AEROS quadratic lsqx5": C_PURPLE,
        "ELVIRA": C_ORANGE,
        "ELVIRA W": C_RED,
    }
    method_name = {
        "AEROS quadratic": "AEROS quadratic",
        "AEROS linear": "AEROS linear",
        "AEROS cubic": "AEROS cubic",
        "AEROS quadratic lsqx5": "AEROS quadratic LSQ",
        "ELVIRA": "ELVIRA",
        "ELVIRA W": "ELVIRA W"
    }

    threshold_hinv = 30

    for shape, sub_df in df.groupby("shape"):
        sub_df["label_plot"] = sub_df.apply(
            lambda x: get_label4plot(x["label"], x["refinement"]),
            axis=1)

        with save_figure(filename=f"Convergence_{shape}", path=experiment_path, figsize=(16, 8), show=False) as (
                fig, ax):
            for (label_plot, label, refinement), df4plot in sub_df.groupby(["label_plot", "label", "refinement"]):
                hinv = df4plot["num_cells_per_dim"].values
                valid_ix = hinv >= threshold_hinv
                rate, origin = np.ravel(np.linalg.lstsq(
                    np.vstack([np.log(hinv[valid_ix]), np.ones(np.sum(valid_ix))]).T,
                    np.log(df4plot["error"].values[valid_ix]).reshape((-1, 1)), rcond=None)[0])
                label_plot_rate = fr"{label_plot}: $\cal{{O}}$({abs(rate):.1f})"
                # ax.scatter(df4plot["num_cells_per_dim"], df4plot["error"],
                #             color=color[label], marker=marker_style[refinement], s=30)
                ax.plot(df4plot["num_cells_per_dim"], df4plot["error"],
                        color=color[label],
                        marker=marker_style[refinement],
                        linestyle=line_style[refinement], linewidth=2, label=label_plot_rate)
                # plot fitting line
                # ax.plot(hinv[valid_ix], np.exp(origin) * hinv[valid_ix] ** rate,
                #          color=color[label], linestyle=line_style[refinement], linewidth=2, label=label_plot_rate)

            # ax = sns.lineplot(sub_df, ax=ax, x="num_cells_per_dim", y="error", hue="label", style="refinement")
            ax.set_xscale("log")
            ax.set_yscale("log")

            xticks = sorted(pd.unique(sub_df["num_cells_per_dim"]))
            ax.set_xlim((int(min(xticks) * 0.9), int(max(xticks) * 1.1)))
            ax.set_xticks(xticks, labels=list(map(str, xticks)))
            ax.grid(True)

            ax.set_title(shape)
            ax.set_xlabel(r"$1/h$", fontdict=axis_font_dict)
            ax.set_ylabel(r"$\|u-\tilde u \|_{L^1}$", fontdict=axis_font_dict)
            ax.legend(prop=legend_font_dict, loc='upper left', bbox_to_anchor=(1, 1))
            ax.tick_params(labelsize=axis_font_dict["size"])
            # ax.set_ylim((1e-7, 1e-1))
            fig.tight_layout()
