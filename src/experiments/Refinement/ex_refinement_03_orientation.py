from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from experiments.OtherExperiments.SubcellExperiments.models2compare import aero_linear
from experiments.Refinement.ex_refinement_config import experiment_subdivision_path, fig_size, cmap_reconstruction, \
    cmap_true_image, curve_color
from experiments.Refinement.ex_refinement_tools import fit_model, obtain_image4error
from experiments.tools import load_image, calculate_averages_from_curve
from experiments.tools4binary_images import plot_reconstruction4img
from perplexitylab.experiment_tools import experiment_iterator, concatenate_iterators, define_default_constants, \
    define_default_variables
from perplexitylab.plot_tools import save_figure
from perplexitylab.experiment_tools import perplexifier
from experiments.Refinement.ex_refinement_tools import efficient_reconstruction
from experiments.Refinement.ex_refinement_01_test_cases import shapes, get_shape_key, shape_names_for_label_plots

# 1680: 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 20, 21, 24, 28, 30, 35, 40, 42, 48, 56, 60, 70, 80, 84, 105, 120, 140, 168, 210, 240, 280, 336, 420, 560, 840, 1680
# divisors = [i for i in range(1, n+1) if n % i == 0]

# --------- PATHs to experiment --------- #
experiment_path = experiment_subdivision_path.joinpath("Orientation")
experiment_path.mkdir(parents=True, exist_ok=True)

plx_fit_model = perplexifier(default_path=experiment_path)(fit_model)
plx_efficient_reconstruction = perplexifier(default_path=experiment_path,
                                            saver=lambda data, filepath: plt.imsave(filepath, data),
                                            loader=lambda filepath: load_image(filepath, other_path=""),
                                            file_format="png")(efficient_reconstruction)

plx_obtain_image4error = perplexifier(default_path=experiment_path,
                                      saver=lambda data, filepath: plt.imsave(filepath, data),
                                      loader=lambda filepath: load_image(filepath, other_path=""),
                                      file_format="png")(obtain_image4error)

if __name__ == "__main__":
    # ---------- Experiment list ---------- #
    iterator_builder, info = experiment_iterator(
        experiment_name=Path(__file__).stem,
        constants=define_default_constants(sub_cell_model=None, label=None, angle_threshold=0, reconstruction_factor=1,
                                           sub_discretization2bound_error=2 * 2 * 3 * 2,
                                           # sub_discretization2bound_error=2 * 2 * 3,
                                           p=1, recalculate_inner_funcs=False, evaluation_mode=False),
        variables=define_default_variables(
            num_cells_per_dim=[40],
            shape=[
                # shapes["Sinusoidal-horizon"],
                shapes["Circle"],
            ],
            refinement=[1, 2, 3]
        )
    )


    def identifier(info):
        return f"Img{get_shape_key(info.shape).split('.')[0]}_{info.num_cells_per_dim}x{info.num_cells_per_dim}_{info.label}_Ref{info.refinement}"


    # ---------- Do experiments ---------- #
    recalculate_all = False

    iterators = []
    # iterators.append(
    #     iterator_builder(sub_cell_model=elvira, label="ELVIRA", refinement=[1, 2, 3], angle_threshold=0,
    #                      recalculate=False or recalculate_all)
    # )
    iterators.append(
        iterator_builder(sub_cell_model=aero_linear, label="AEROS linear", angle_threshold=45,
                         recalculate=False or recalculate_all)
    )

    for experiment_info in concatenate_iterators(*iterators)():
        print("----------------------------------")
        print(identifier(experiment_info))
        avg_values = calculate_averages_from_curve(experiment_info.shape, (experiment_info.num_cells_per_dim,
                                                                           experiment_info.num_cells_per_dim))
        true_reconstruction = plx_obtain_image4error(
            shape=experiment_info.shape, num_cells_per_dim=experiment_info.num_cells_per_dim,
            sub_discretization2bound_error=experiment_info.sub_discretization2bound_error,
            avg_values=avg_values, evaluation_mode=experiment_info.evaluation_mode)
        model = plx_fit_model(recalculate=experiment_info.recalculate_inner_funcs,
                              sub_cell_model=experiment_info.sub_cell_model,
                              angle_threshold=experiment_info.angle_threshold,
                              refinement=experiment_info.refinement, avg_values=avg_values)
        reconstruction = plx_efficient_reconstruction(
            model=model, avg_values=avg_values,
            sub_discretization2bound_error=experiment_info.sub_discretization2bound_error,
            refinement=experiment_info.refinement, evaluation_mode=experiment_info.evaluation_mode,
            recalculate=experiment_info.recalculate_inner_funcs)

        shape_name = get_shape_key(experiment_info.shape)
        for name, plot_curve, plot_vh_classification in [("VH_classification", False, True), ("Curve", True, False)]:
            with save_figure(filename=f"{name}_{identifier(experiment_info)}", path=experiment_path, figsize=fig_size,
                             show=False) as (fig, ax):
                plot_reconstruction4img(
                    fig=fig, ax=ax,
                    image=true_reconstruction,
                    num_cells_per_dim=experiment_info.num_cells_per_dim,
                    model=model,
                    reconstruction=reconstruction,
                    difference=False,
                    plot_curve=plot_curve,
                    plot_curve_winner=False,
                    plot_vh_classification=plot_vh_classification,
                    plot_singular_cells=False,
                    alpha_true_image=0.15,
                    alpha=0,
                    # trim=((1, 1), (2, 2)),
                    # trim=np.array(((10, 9), (9, 10))) * experiment_info.refinement,
                    trim=np.array(((16, 9), (15, 10))) * 2 ** experiment_info.refinement,
                    cmap=cmap_reconstruction,
                    cmap_true_image=cmap_true_image,
                    curve_color=curve_color,
                    vmin=0, vmax=1,
                    labels=False,
                    draw_mesh=True,
                    numbers_on=True,
                )
