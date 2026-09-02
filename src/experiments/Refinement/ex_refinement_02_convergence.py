import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.Refinement.ex_refinement_01_test_cases import shapes, get_shape_key, shape_names_for_label_plots
from experiments.Refinement.ex_refinement_config import experiment_subdivision_path, experiment_name, axis_font_dict, \
    legend_font_dict, line_style, marker_style, color, experiment_labels_order
from experiments.Refinement.ex_refinement_models_to_compare import quadratic, aero_linear, elvira, cubic, \
    quartic
from experiments.Refinement.ex_refinement_tools import fit_model, calculate_error, \
    efficient_reconstruction, obtain_image4error
from experiments.tools import calculate_averages_from_curve
from perplexitylab.experiment_tools import experiment_iterator, concatenate_iterators, define_default_constants, \
    define_default_variables, perplexifier, do
from perplexitylab.plot_tools import save_figure, sorted_legend

# --------- PATHs to experiment --------- #
experiment_path = experiment_subdivision_path.joinpath("Convergence")
experiment_path.mkdir(parents=True, exist_ok=True)


# --------- Auxiliary functions --------- #
def identifier(experiment_info):
    return f"Img{experiment_info.shape}_{experiment_info.num_cells_per_dim}x{experiment_info.num_cells_per_dim}_{experiment_info.label}_Ref{experiment_info.refinement}"


def get_label4plot(label, refinement):
    return f'{label}{" Subdivisions=" + str(refinement - 1) if refinement > 1 else ""}'


@perplexifier(default_path=experiment_path)
def experiment_convergence(shape, sub_cell_model, refinement, angle_threshold, num_cells_per_dim,
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
    return error


if __name__ == "__main__":
    # ---------- Experiment variables and constants default ---------- #
    iterator_builder, info = experiment_iterator(
        experiment_name=Path(__file__).stem,
        constants=define_default_constants(sub_cell_model=None, label=None, angle_threshold=0, reconstruction_factor=1,
                                           sub_discretization2bound_error=2 * 2 * 3 * 2,
                                           # sub_discretization2bound_error=2 * 2 * 3,
                                           p=1, recalculate_inner_funcs=False, evaluation_mode=False),
        variables=define_default_variables(
            num_cells_per_dim=[20, 30, 40, 50, 60, 70, 80, 90, 100],
            # num_cells_per_dim=[20, 50, 100],
            shape=[
                shapes["Sinusoidal-horizon"],
                shapes["Circle"],
            ],
            refinement=[1, 2, ]
        )
    )

    # ---------- Do experiments ---------- #
    recalculate_all = False

    iterators = []
    iterators.append(
        iterator_builder(sub_cell_model=elvira, label="ELVIRA", refinement=[1, 2, 3], angle_threshold=0,
                         recalculate=False or recalculate_all)
    )
    iterators.append(
        iterator_builder(sub_cell_model=aero_linear, label="AEROS linear", refinement=[1, 2, 3], angle_threshold=45,
                         recalculate=False or recalculate_all)
    )
    iterators.append(
        iterator_builder(sub_cell_model=quadratic, label="AEROS quadratic", refinement=[1, ], angle_threshold=45,
                         recalculate=False or recalculate_all)
    )
    iterators.append(
        iterator_builder(sub_cell_model=quadratic, label="AEROS quadratic", refinement=[2, ], angle_threshold=45,
                         num_cells_per_dim=[30, 40, 50, 60, 70, 80, 90, 100],
                         recalculate=False or recalculate_all)
    )
    iterators.append(
        iterator_builder(sub_cell_model=cubic, label="AEROS cubic", refinement=[1, 2], angle_threshold=45,
                         recalculate=False or recalculate_all)
    )
    iterators.append(
        iterator_builder(sub_cell_model=quartic, label="AEROS quartic", refinement=[1, 2], angle_threshold=0,
                         num_cells_per_dim=[30, 40, 50, 60, 70, 80, 90, 100],
                         recalculate=False or recalculate_all)
    )

    data = defaultdict(list)
    for experiment_info in concatenate_iterators(*iterators)():
        print("\n----------------------------------")
        print(identifier(experiment_info))
        error = do(experiment_convergence, experiment_info)
        data["error"].append(error)
        data["label"].append(experiment_info.label)
        data["refinement"].append(experiment_info.refinement)
        data["num_cells_per_dim"].append(experiment_info.num_cells_per_dim)
        data["shape_name"].append(get_shape_key(experiment_info.shape))
    df = pd.DataFrame.from_dict(data)

    # ---------- Do plot ---------- #
    threshold_hinv = 30

    for shape_name, sub_df in df.groupby("shape_name"):
        sub_df["label_plot"] = sub_df.apply(
            lambda x: get_label4plot(x["label"], x["refinement"]),
            axis=1)

        convergence_rates = defaultdict(list)
        labels_order = dict()
        with save_figure(filename=f"Convergence_{shape_name}", path=experiment_path, figsize=(16, 8), show=False) as (
                fig, ax):
            for (label_plot, label, refinement), df4plot in sub_df.groupby(["label_plot", "label", "refinement"]):
                hinv = df4plot["num_cells_per_dim"].values
                valid_ix = hinv >= threshold_hinv
                rate, origin = np.ravel(np.linalg.lstsq(
                    np.vstack([np.log(hinv[valid_ix]), np.ones(np.sum(valid_ix))]).T,
                    np.log(df4plot["error"].values[valid_ix]).reshape((-1, 1)), rcond=None)[0])

                convergence_rates["rate"].append(rate)
                convergence_rates["label"].append(label)
                convergence_rates["subdivisions"].append(refinement - 1)

                # label_plot_rate = fr"{label_plot}: $\cal{{O}}$({abs(rate):.1f})"
                # label_plot_rate = fr"{label_plot}" if refinement == 0 else None
                label_plot_rate = fr"{label_plot}"

                # replaces with the plot label
                labels_order[label_plot_rate] = 100 * experiment_labels_order.index(label) + refinement
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

            ax.set_title(shape_names_for_label_plots[shape_name])
            ax.set_xlabel(r"$1/h$", fontdict=axis_font_dict)
            # ax.set_ylabel(r"$\|u-\tilde u \|_{L^1}$", fontdict=axis_font_dict)
            ax.set_ylabel(r"$L^1$ error", fontdict=axis_font_dict)
            ax.legend(prop=legend_font_dict, loc='upper left', bbox_to_anchor=(1, 1))
            sorted_legend(ax, labels_order, prop=legend_font_dict, loc='upper left', bbox_to_anchor=(1, 1))
            ax.tick_params(labelsize=axis_font_dict["size"])
            # ax.set_ylim((1e-7, 1e-1))
            fig.tight_layout()

        pd.DataFrame.from_dict(convergence_rates).to_csv(f"{experiment_path}/{shape_name}_convergence_rates.csv")
