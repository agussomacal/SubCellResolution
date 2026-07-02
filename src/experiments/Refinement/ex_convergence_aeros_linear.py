from collections import defaultdict
from collections import defaultdict
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from experiments.Refinement.ex_refinement_config import experiment_path, experiment_name, C_GREEN, C_BLUE, C_PURPLE
from experiments.Refinement.ex_refinement_convergence import single_experiment_convergence
from experiments.Refinement.ex_refinement_models_to_compare import aero_linear, aero_linear_w
from lib.Curves.CurveCircle import CurveCircle, CircleParams
from perplexitylab.experiment_tools import experiment_iterator, concatenate_iterators, define_default_constants, \
    define_default_variables, perplexifier
from perplexitylab.miscellaneous import filter_for_func, plx_partial
from perplexitylab.plot_tools import save_figure

file_format_data_to_plot = "csv"
filename_data_to_plot = "AEROSLinearConvergencePlot"
path_data_to_plot = f"{experiment_path}/{experiment_name}"

# Experiment general params
recalculate_obtain_image4error = False


def identifier(experiment_info):
    return f"Img{experiment_info.shape}_{experiment_info.num_cells_per_dim}x{experiment_info.num_cells_per_dim}_AngleThreshold{experiment_info.angle_threshold}_Ref{experiment_info.refinement}"


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
            recalculate=experiment_info.recalculate, sub_cell_model=plx_partial(aero_linear_w, ccew=experiment_info.ccew),
            **filter_for_func(single_experiment_convergence, experiment_info._asdict())
        )
        data["error"].append(error)
        data["angle_threshold"].append(experiment_info.angle_threshold)
        data["ccew"].append(experiment_info.ccew)
        data["refinement"].append(experiment_info.refinement)
        data["num_cells_per_dim"].append(experiment_info.num_cells_per_dim)
        data["shape"].append(str(experiment_info.shape))
    return pd.DataFrame.from_dict(data)


if __name__ == "__main__":
    # ---------- Experiment variables and constants default ---------- #
    iterator_builder, info = experiment_iterator(
        experiment_name=Path(__file__).stem,
        constants=define_default_constants(reconstruction_factor=1,
                                           sub_discretization2bound_error=6, p=1, recalculate=True,
                                           evaluation_mode=False),
        variables=define_default_variables(
            # num_cells_per_dim=[10, 20, 30, 40, 50, 60, 70, 80, 90],
            # num_cells_per_dim=[20, 30, 40, 50, 60, 70, 80, 90, 100],
            # num_cells_per_dim=[20, 30, 40, ],
            num_cells_per_dim=[50, 60, 70, 80, 90, 100],
            shape=[CurveCircle(params=CircleParams(x0=0.511, y0=0.486, radius=0.232))],
            refinement=[1, 2],
            # angle_threshold=[0, 27.5, 45],
            angle_threshold=[0, 45],
            ccew=[10000],
        ))

    # ---------- Do experiments ---------- #
    _, df = do_experiment_convergence(
        recalculate=True,
        iterators=(iterator_builder(),),
    )

    # ---------- Do plot ---------- #
    axis_font_dict = {'color': 'black', 'weight': 'normal', 'size': 25}
    labels_font_dict = {'color': 'black', 'weight': 'normal', 'size': 25}
    legend_font_dict = {'weight': 'normal', "size": 19, 'stretch': 'normal'}
    line_style = {0: "solid", 27.5: "dashed", 45: "dashdot"}
    marker_style = {0: "o", 27.5: "^", 45: "s"}
    color = {
        0: C_GREEN,
        10000: C_BLUE,
    }

    threshold_hinv = 50

    for shape, sub_df in df.groupby("shape"):
        sub_df["label_plot"] = sub_df.apply(
            lambda
                x: f'Central cell weight={x["ccew"]} - Angle threshold={x["angle_threshold"]} Ref{x["refinement"]}',
            axis=1)

        with save_figure(filename=f"AEROSLinear_Convergence_{shape}", path=experiment_path, figsize=(12, 8), show=False) as (
                fig, ax):
            for (label_plot, angle_threshold, ccew), df4plot in sub_df.groupby(
                    ["label_plot", "angle_threshold", "ccew"]):
                hinv = df4plot["num_cells_per_dim"].values
                valid_ix = hinv >= threshold_hinv
                rate, origin = np.ravel(np.linalg.lstsq(
                    np.vstack([np.log(hinv[valid_ix]), np.ones(np.sum(valid_ix))]).T,
                    np.log(df4plot["error"].values[valid_ix]).reshape((-1, 1)), rcond=None)[0])
                label_plot_rate = fr"{label_plot}: $\cal{{O}}$({abs(rate):.1f})"
                plt.plot(df4plot["num_cells_per_dim"], df4plot["error"], label=label_plot_rate,
                         linestyle=line_style[angle_threshold], color=color[ccew], linewidth=2,
                         marker=marker_style[angle_threshold])
                plt.plot(hinv[valid_ix], np.exp(origin) * hinv[valid_ix] ** rate,
                         color="black", linestyle="solid", linewidth=1, )

            # ax = sns.lineplot(sub_df, ax=ax, x="num_cells_per_dim", y="error", hue="label", style="refinement")
            ax.set_title("AEROS linear angle threshold behaviour")
            ax.set_xscale("log")
            ax.set_yscale("log")

            xticks = sorted(pd.unique(sub_df["num_cells_per_dim"]))
            ax.set_xlim((int(min(xticks) * 0.8), int(max(xticks) * 1.2)))
            ax.set_xticks(xticks, labels=list(map(str, xticks)))
            ax.grid(True)

            ax.set_title(shape)
            ax.set_xlabel(r"$1/h$", fontdict=axis_font_dict)
            ax.set_ylabel(r"$\|u-\tilde u \|_{L^1}$", fontdict=axis_font_dict)
            ax.legend(prop=legend_font_dict)
            ax.tick_params(labelsize=axis_font_dict["size"])
            # ax.set_ylim((1e-7, 1e-1))
            fig.tight_layout()
