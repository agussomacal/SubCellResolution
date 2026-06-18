from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from experiments.Refinement.ex_refinement_config import experiment_path, fig_size
from experiments.Refinement.ex_refinement_tools import fit_model, calculate_error
from experiments.Refinement.models_to_compare import quadratic, aero_linear
from experiments.tools import load_image, calculate_averages_from_curve, singular_cells_mask, make_image_high_resolution
from lib.Curves.CurveCircle import CurveCircle, CircleParams
from lib.SubCellReconstruction import reconstruct_by_factor
from perplexitylab.experiment_tools import experiment_iterator, concatenate_iterators, define_default_constants, \
    define_default_variables, perplexifier
from perplexitylab.plot_tools import save_figure


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


if __name__ == "__main__":
    # Experiment general params
    noise = 0
    seed = 42
    recalculate_all = False

    # ---------- Experiment list ---------- #
    iterator_builder, info = experiment_iterator(
        experiment_name=Path(__file__).stem,
        constants=define_default_constants(sub_cell_model=None, label=None, angle_threshold=0, reconstruction_factor=1,
                                           sub_discretization2bound_error=18, p=1, recalculate=recalculate_all),
        variables=define_default_variables(
            num_cells_per_dim=[10, 20, 30, 40, 50, 60, 70, 80],
            shape=[CurveCircle(params=CircleParams(x0=0.511, y0=0.486, radius=0.232))],
            # image_name=["batata.jpg"],
            refinement=[1, 2]
        ))


    def identifier(info):
        return f"Img{info.shape}_{info.num_cells_per_dim}x{info.num_cells_per_dim}_{info.label}_Ref{info.refinement}"


    iterators = concatenate_iterators(
        iterator_builder(sub_cell_model=quadratic, label="AEROS quadratic", refinement=[1],
                         recalculate=False or recalculate_all),
        iterator_builder(sub_cell_model=aero_linear, label="AEROS linear", refinement=[1, 2],
                         recalculate=False or recalculate_all),
        # iterator_builder(sub_cell_model=elvira, label="ELVIRA", recalculate=False or recalculate_all),
    )

    # ---------- Do experiments ---------- #
    data = defaultdict(list)


    def gather(data, info, error):
        data["error"].append(error)
        data["label"].append(info.label)
        data["refinement"].append(info.refinement)
        data["num_cells_per_dim"].append(info.num_cells_per_dim)
        data["shape"].append(str(info.shape))
        return data


    for experiment_info in iterators():
        print("----------------------------------")
        print(identifier(experiment_info))
        avg_values = calculate_averages_from_curve(experiment_info.shape, (experiment_info.num_cells_per_dim,
                                                                           experiment_info.num_cells_per_dim))

        hash_value = 42
        hash_value, model = fit_model(
            hash_of_preprocess=hash_value, recalculate=experiment_info.recalculate,
            sub_cell_model=experiment_info.sub_cell_model,
            angle_threshold=experiment_info.angle_threshold,
            refinement=experiment_info.refinement, avg_values=avg_values)
        hash_value, true_reconstruction = obtain_image4error(
            hash_of_preprocess=hash_value, recalculate=experiment_info.recalculate,
            shape=experiment_info.shape, num_cells_per_dim=experiment_info.num_cells_per_dim,
            sub_discretization2bound_error=experiment_info.sub_discretization2bound_error,
            avg_values=avg_values)
        hash_value, reconstruction = efficient_reconstruction(
            hash_of_preprocess=hash_value, recalculate=experiment_info.recalculate,
            model=model, avg_values=avg_values,
            sub_discretization2bound_error=experiment_info.sub_discretization2bound_error,
            refinement=experiment_info.refinement)
        error = calculate_error(true_reconstruction, reconstruction, p=experiment_info.p)
        data = gather(data, experiment_info, error=error)

    df = pd.DataFrame.from_dict(data)
    for shape, sub_df in df.groupby(["shape"]):
        with save_figure(filename=f"Convergence_{shape}", path=experiment_path, figsize=fig_size, show=False) as (
                fig, ax):
            ax = sns.lineplot(sub_df, ax=ax, x="num_cells_per_dim", y="error", hue="label", style="refinement")
            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.set_xticks(ax.get_xticks(), visible=False)
            xticks = sorted(pd.unique(sub_df["num_cells_per_dim"]))
            ax.set_xlim((int(min(xticks) * 0.8), int(max(xticks) * 1.2)))
            ax.set_xticks(xticks, labels=list(map(str, xticks)))

            ax.set_title(shape)
            ax.set_xlabel(r"$1/h$")
            ax.set_ylabel(r"$\|u-v\|$")
            fig.tight_layout()
