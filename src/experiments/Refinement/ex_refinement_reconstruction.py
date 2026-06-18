from pathlib import Path

from experiments.OtherExperiments.SubcellExperiments.models2compare import quadratic, aero_linear, elvira
from experiments.Refinement.ex_refinement_config import experiment_path, curve_color, cmap_reconstruction, \
    cmap_true_image, fig_size
from experiments.Refinement.ex_refinement_tools import image_to_avg, do_reconstruction, fit_model
from experiments.tools import load_image
from experiments.tools4binary_images import plot_reconstruction4img
from perplexitylab.experiment_tools import experiment_iterator, concatenate_iterators, define_default_constants, \
    define_default_variables
from perplexitylab.plot_tools import save_figure

if __name__ == "__main__":
    # Experiment general params
    noise = 0
    seed = 42
    recalculate_all = False

    # ---------- Experiment list ---------- #
    iterator_builder, info = experiment_iterator(
        experiment_name=Path(__file__).stem,
        constants=define_default_constants(sub_cell_model=None, label=None, angle_threshold=0, reconstruction_factor=1,
                                           recalculate=recalculate_all),
        variables=define_default_variables(
            num_cells_per_dim=[20, 42],
            image_name=["batata.jpg"],
            refinement=[1, 2, 3]
        ))


    def identifier(info):
        return f"Img{info.image_name.split('.')[0]}_{info.num_cells_per_dim}x{info.num_cells_per_dim}_{info.label}_Ref{info.refinement}"


    iterators = concatenate_iterators(
        iterator_builder(sub_cell_model=quadratic, label="AEROS quadratic", recalculate=False or recalculate_all),
        iterator_builder(sub_cell_model=aero_linear, label="AEROS linear", recalculate=False or recalculate_all),
        iterator_builder(sub_cell_model=elvira, label="ELVIRA", recalculate=False or recalculate_all),
    )

    # ---------- Do experiments ---------- #
    for experiment_info in iterators():
        print("----------------------------------")
        print(identifier(experiment_info))
        image = load_image(experiment_info.image_name)
        avg_values = image_to_avg(num_cells_per_dim=experiment_info.num_cells_per_dim, image=image, noise=noise,
                                  seed=seed)
        hash_value = 42
        hash_value, model = fit_model(hash_of_preprocess=hash_value, recalculate=experiment_info.recalculate,
                                      sub_cell_model=experiment_info.sub_cell_model,
                                      angle_threshold=experiment_info.angle_threshold,
                                      refinement=experiment_info.refinement, avg_values=avg_values)
        hash_value, reconstruction = do_reconstruction(hash_of_preprocess=hash_value,
                                                       recalculate=experiment_info.recalculate,
                                                       image=image, model=model,
                                                       reconstruction_factor=experiment_info.reconstruction_factor)
        with save_figure(filename=identifier(experiment_info), path=experiment_path, figsize=fig_size,
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
