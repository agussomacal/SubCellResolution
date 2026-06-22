import warnings
from collections import defaultdict
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from experiments.Refinement.ex_refinement_config import experiment_path, experiment_name, C_GREEN, C_BLUE, C_PURPLE, \
    C_ORANGE, C_RED
from experiments.Refinement.ex_refinement_models_to_compare import quadratic, aero_linear, elvira, \
    elvira_w, aero_linear_w
from experiments.Refinement.ex_refinement_tools import fit_model, calculate_error
from experiments.tools import load_image, calculate_averages_from_curve, singular_cells_mask, make_image_high_resolution
from lib.Curves.CurveCircle import CurveCircle, CircleParams
from lib.SubCellReconstruction import reconstruct_by_factor
from perplexitylab.experiment_tools import experiment_iterator, concatenate_iterators, define_default_constants, \
    define_default_variables, perplexifier
from perplexitylab.miscellaneous import filter_for_func
from perplexitylab.plot_tools import save_figure

file_format_data_to_plot = "csv"
filename_data_to_plot = "ConvergencePlot"
path_data_to_plot = f"{experiment_path}/{experiment_name}"


def identifier(experiment_info):
    return f"Img{experiment_info.shape}_{experiment_info.num_cells_per_dim}x{experiment_info.num_cells_per_dim}_{experiment_info.label}_Ref{experiment_info.refinement}"


@perplexifier(default_path=experiment_path,
              saver=lambda data, filepath: plt.imsave(filepath, data),
              loader=lambda filepath: load_image(filepath, other_path=""),
              file_format="png")
def obtain_image4error(shape, num_cells_per_dim, sub_discretization2bound_error, avg_values):
    edge_mask = make_image_high_resolution(singular_cells_mask(avg_values),
                                           reconstruction_factor=sub_discretization2bound_error)
    cells2reconstruct = list(zip(*np.where(edge_mask)))
    true_reconstruction = make_image_high_resolution(avg_values, reconstruction_factor=sub_discretization2bound_error)
    true_reconstruction[edge_mask] = calculate_averages_from_curve(
        shape,
        (num_cells_per_dim * sub_discretization2bound_error,
         num_cells_per_dim * sub_discretization2bound_error),
        cells2reconstruct=cells2reconstruct)[edge_mask]

    return true_reconstruction


@perplexifier(default_path=experiment_path,
              saver=lambda data, filepath: plt.imsave(filepath, data),
              loader=lambda filepath: load_image(filepath, other_path=""),
              file_format="png")
def efficient_reconstruction(model, avg_values, sub_discretization2bound_error, refinement):
    """
    Only reconstructs fully in the cells where there is discontinuity otherwise copies avgcells values
    :return:
    """

    edge_mask = singular_cells_mask(avg_values)
    edge_mask = np.repeat(np.repeat(edge_mask, refinement, axis=0), refinement, axis=1)
    cells2reconstruct = list(zip(*np.where(edge_mask)))

    reconstruction = np.repeat(np.repeat(avg_values, sub_discretization2bound_error, axis=0),
                               sub_discretization2bound_error, axis=1)

    magnification = sub_discretization2bound_error // refinement
    edge_mask_hr = np.repeat(np.repeat(edge_mask, magnification, axis=0), magnification, axis=1)
    reconstruction[edge_mask_hr] = \
        reconstruct_by_factor(cells=model.cells, resolution=model.resolution, cells2reconstruct=cells2reconstruct,
                              resolution_factor=magnification)[edge_mask_hr]
    return reconstruction


@perplexifier(default_path=experiment_path)
def single_experiment_convergence(shape, sub_cell_model, refinement, angle_threshold, num_cells_per_dim,
                                  sub_discretization2bound_error, p, recalculate_inner_funcs, hash_value=42):
    avg_values = calculate_averages_from_curve(shape, (num_cells_per_dim, num_cells_per_dim))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hash_value, model = fit_model(
            hash_of_preprocess=hash_value, recalculate=recalculate_inner_funcs,
            sub_cell_model=sub_cell_model,
            angle_threshold=angle_threshold,
            refinement=refinement, avg_values=avg_values)
    hash_value, true_reconstruction = obtain_image4error(
        hash_of_preprocess=hash_value, recalculate=recalculate_inner_funcs,
        shape=shape, num_cells_per_dim=num_cells_per_dim,
        sub_discretization2bound_error=sub_discretization2bound_error,
        avg_values=avg_values)
    hash_value, reconstruction = efficient_reconstruction(
        hash_of_preprocess=hash_value, recalculate=recalculate_inner_funcs,
        model=model, avg_values=avg_values,
        sub_discretization2bound_error=sub_discretization2bound_error,
        refinement=refinement)
    error = calculate_error(true_reconstruction, reconstruction, p=p)
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
        _, error = single_experiment_convergence(recalculate=experiment_info.recalculate,
                                                 **filter_for_func(single_experiment_convergence,
                                                                   experiment_info._asdict()))
        data["error"].append(error)
        data["label"].append(experiment_info.label)
        data["refinement"].append(experiment_info.refinement)
        data["num_cells_per_dim"].append(experiment_info.num_cells_per_dim)
        data["shape"].append(str(experiment_info.shape))
    return pd.DataFrame.from_dict(data)


if __name__ == "__main__":
    # Experiment general params
    recalculate_all = False

    # ---------- Experiment variables and constants default ---------- #
    iterator_builder, info = experiment_iterator(
        experiment_name=Path(__file__).stem,
        constants=define_default_constants(sub_cell_model=None, label=None, angle_threshold=0, reconstruction_factor=1,
                                           sub_discretization2bound_error=18, p=1, recalculate=False,
                                           recalculate_inner_funcs=False),
        variables=define_default_variables(
            # num_cells_per_dim=[10, 20, 30, 40, 50, 60, 70, 80, 90],
            num_cells_per_dim=[10, 20, 30, 40, 50, 60, 70, 80, 90],
            shape=[CurveCircle(params=CircleParams(x0=0.511, y0=0.486, radius=0.232))],
            # image_name=["batata.jpg"],
            refinement=[1, 2]
        ))

    # ---------- Do experiments ---------- #
    _, df = do_experiment_convergence(
        recalculate=True,
        iterators=(
            iterator_builder(sub_cell_model=quadratic, label="AEROS quadratic", refinement=[1],
                             recalculate=False or recalculate_all),
            iterator_builder(sub_cell_model=aero_linear, label="AEROS linear", refinement=[1, 2],
                             recalculate=False or recalculate_all, recalculate_inner_funcs=False),
            # iterator_builder(sub_cell_model=aero_linear_w, label="AEROS linear W", refinement=[1, 2],
            #                  recalculate=False or recalculate_all, recalculate_inner_funcs=False),
            # iterator_builder(sub_cell_model=elvira, label="ELVIRA", refinement=[1, 2],
            #                  recalculate=False or recalculate_all, recalculate_inner_funcs=False),
            # iterator_builder(sub_cell_model=elvira_w, label="ELVIRA W", refinement=[1, 2],
            #                  recalculate=False or recalculate_all, recalculate_inner_funcs=False),
        ),
    )

    # ---------- Do plot ---------- #
    axis_font_dict = {'color': 'black', 'weight': 'normal', 'size': 25}
    labels_font_dict = {'color': 'black', 'weight': 'normal', 'size': 25}
    legend_font_dict = {'weight': 'normal', "size": 19, 'stretch': 'normal'}
    line_style = {1: "solid", 2: "dashed", 3: "dashdot", 4: "dotted"}
    marker_style = {1: ".", 2: "^", 3: "", 4: ""}
    color = {
        "AEROS quadratic": C_GREEN,
        "AEROS linear": C_BLUE,
        "AEROS linear W": C_PURPLE,
        "ELVIRA": C_ORANGE,
        "ELVIRA W": C_RED,
    }
    method_name = {
        "AEROS quadratic": "AEROS quadratic",
        "AEROS linear": "AEROS linear",
        "AEROS linear W": "AEROS linear W",
        "ELVIRA": "ELVIRA",
        "ELVIRA W": "ELVIRA W"
    }

    threshold_hinv = 50

    for shape, sub_df in df.groupby("shape"):
        sub_df["label_plot"] = sub_df.apply(
            lambda x: f'{x["label"]}{" Subdivisions=" + str(x["refinement"] - 1) if x["refinement"] > 1 else ""}',
            axis=1)

        with save_figure(filename=f"Convergence_{shape}", path=experiment_path, figsize=(12, 8), show=False) as (
                fig, ax):
            for (label_plot, label, refinement), df4plot in sub_df.groupby(["label_plot", "label", "refinement"]):
                hinv = df4plot["num_cells_per_dim"].values
                valid_ix = hinv > threshold_hinv
                rate, origin = np.ravel(np.linalg.lstsq(
                    np.vstack([np.log(hinv[valid_ix]), np.ones(np.sum(valid_ix))]).T,
                    np.log(df4plot["error"].values[valid_ix]).reshape((-1, 1)), rcond=None)[0])
                label_plot_rate = fr"{label_plot}: $\cal{{O}}$({abs(rate):.1f})"
                plt.plot(df4plot["num_cells_per_dim"], df4plot["error"], label=label_plot_rate,
                         linestyle=line_style[refinement], color=color[label], linewidth=2,
                         marker=marker_style[refinement])
                plt.plot(hinv[valid_ix], np.exp(origin) * hinv[valid_ix] ** rate,
                         color="black", linestyle="solid", linewidth=1,)

            # ax = sns.lineplot(sub_df, ax=ax, x="num_cells_per_dim", y="error", hue="label", style="refinement")
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
