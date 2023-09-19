import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.visualization import generic_plot
from experiments.subcell_paper.models2compare import aero_linear, aero_lq_vertex, elvira, quadratic, winner_color_dict
from experiments.subcell_paper.tools4binary_images import fit_model, plot_reconstruction

if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='Refinement',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )

    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "models",
        *map(fit_model, [
            elvira,
            quadratic,
        ]),
        recalculate=False
    )
    lab.execute(
        data_manager,
        num_cores=15,
        forget=False,
        save_on_iteration=None,
        refinement=[1, 2],
        num_cells_per_dim=[20, 42],  # 20, 42, 84 168 , 84
        noise=[0],
        image=[
            # "yoda.jpg",
            # "DarthVader.jpeg",
            # "Ellipsoid_1680x1680.jpg",
            "ShapesVertex_1680x1680.jpg",
            # "HandVertex_1680x1680.jpg",
            # "Polygon_1680x1680.jpg",
        ],
        reconstruction_factor=[5],
        angle_threshold=[20]
    )

    # generic_plot(
    #     data_manager,
    #     x="angle_threshold", y="reconstruction_error", label="models",
    #     plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
    #     # log="y",
    #     axes_by=["image"],
    #     plot_by=["num_cells_per_dim"]
    # )
    #
    # generic_plot(
    #     data_manager,
    #     x="time_to_fit", y="reconstruction_error", label="models",
    #     plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
    #     # log="y",
    #     axes_by=["image"],
    #     plot_by=["num_cells_per_dim"]
    # )

    plot_reconstruction(
        data_manager,
        name="Reconstruction",
        folder='Reconstruction',
        plot_by=['image', 'models', 'num_cells_per_dim', "refinement"],
        # models=[
        #     "aero_lq_vertex"
        # ],
        # image=[
        #     # "yoda.jpg",
        #     # "DarthVader.jpeg",
        #     # "Ellipsoid_1680x1680.jpg",
        #     "ShapesVertex_1680x1680.jpg",
        #     # "HandVertex_1680x1680.jpg",
        #     # "Polygon_1680x1680.jpg",
        # ],
        axes_xy_proportions=(15, 15),
        difference=False,
        plot_curve=True,
        plot_curve_winner=False,
        plot_vh_classification=False,
        plot_singular_cells=False,
        plot_original_image=True,
        numbers_on=True,
        plot_again=True,
        num_cores=1,
        winner_color_dict=winner_color_dict,
        cmap="plasma"
    )
