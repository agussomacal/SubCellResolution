from experiments.OtherExperiments.SubcellExperiments.models2compare import quadratic
from experiments.Refinement.ex_refinement_config import experiment_path, curve_color, cmap_reconstruction, \
    cmap_true_image, fig_size
from experiments.Refinement.ex_refinement_tools import ExperimentInfo, image_to_avg, do_reconstruction, \
    fit_model
from perplexitylab.experiment_tools import experiment_iterator, concatenate_iterators
from experiments.tools import load_image
from experiments.tools4binary_images import plot_reconstruction4img
from perplexitylab.plot_tools import save_figure

if __name__ == "__main__":

    # Experiment general params
    noise = 0
    seed = 42
    recalculate_all = False

    # ---------- Experiment list ---------- #
    common_iterators = {"num_cells_per_dim": [20], "image_name": ["batata.jpg"], "refinement": [1, 2]}
    iterator = concatenate_iterators(
        experiment_iterator(ExperimentInfo, sub_cell_model=quadratic, label="quadratic", recalculate=False)(
            **common_iterators),
    )

    # ---------- Do experiments ---------- #
    for experiment_info, other_variables in iterator():
        if recalculate_all or experiment_info.recalculate:
            print("----------------------------------")
            print(experiment_info.identifier.replace("_", " "))
            image = load_image(experiment_info.image_name)
            avg_values = image_to_avg(experiment_info=experiment_info, image=image, noise=noise,
                                      seed=seed)
            model = fit_model(experiment_info=experiment_info, avg_values=avg_values)
            reconstruction = do_reconstruction(experiment_info=experiment_info, image=image, model=model)

            with save_figure(filename=experiment_info.identifier, path=experiment_path, figsize=fig_size,
                             show=False) as (fig, ax):
                plot_reconstruction4img(
                    fig=fig, ax=ax,
                    image=experiment_info.image_name,
                    num_cells_per_dim=experiment_info.num_cells_per_dim,
                    model=model,
                    reconstruction=reconstruction,
                    difference=False,
                    plot_curve=True,
                    plot_curve_winner=False,
                    plot_vh_classification=False,
                    plot_singular_cells=False,
                    alpha_true_image=0.15,
                    alpha=0,
                    trim=((1, 1), (2, 2)),
                    cmap=cmap_reconstruction,
                    cmap_true_image=cmap_true_image,
                    curve_color=curve_color,
                    vmin=0, vmax=1,
                    labels=False,
                    draw_mesh=False,
                    numbers_on=False,
                )
