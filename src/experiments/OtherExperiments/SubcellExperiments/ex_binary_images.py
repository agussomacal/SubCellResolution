import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from experiments.OtherExperiments.SubcellExperiments.models2compare import aero_linear, aero_qelvira_vertex, elvira, quadratic, winner_color_dict
from experiments.tools4binary_images import fit_model, plot_reconstruction

if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='Vertices',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )

    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "models",
        *map(fit_model, [
            elvira,
            aero_linear,
            quadratic,
            aero_qelvira_vertex,
        ]),
        recalculate=False
    )
    lab.execute(
        data_manager,
        num_cores=15,
        forget=False,
        save_on_iteration=None,
        refinement=[1],
        num_cells_per_dim=[20, 42],  # 20, 42, 84 168 , 84 4220,, 42
        noise=[0],
        image=[
            "yoda.jpg",
            "DarthVader.jpeg",
            "Ellipsoid_1680x1680.jpg",
            "ShapesVertex_1680x1680.jpg",
            "HandVertex_1680x1680.jpg",
            "Polygon_1680x1680.jpg",
        ],
        reconstruction_factor=[5],
        angle_threshold=[25]
    )

    plot_reconstruction(
        data_manager,
        path=config.subcell_paper_figures_path,
        format=".pdf",
        name="Reconstruction",
        folder='Reconstruction',
        axes_by=[],
        plot_by=['image', 'models', 'num_cells_per_dim'],
        models=["aero_qelvira_vertex"],
        # models=["aero_lq_vertex"],
        image=[
            "ShapesVertex_1680x1680.jpg",
            "HandVertex_1680x1680.jpg",
            "Polygon_1680x1680.jpg",
        ],
        axes_xy_proportions=(15, 15),
        difference=False,
        plot_curve=True,
        plot_curve_winner=False,
        plot_vh_classification=False,
        plot_singular_cells=False,
        plot_original_image=True,
        numbers_on=True,
        plot_again=True,
        num_cores=15,
        # cmap="YlOrBr",
        # cmapoi="YlGn",
        # alphaio=0.7,
        # alpha=0,
        winner_color_dict=winner_color_dict,
        draw_cells=False
    )

    plot_reconstruction(
        data_manager,
        name="Reconstruction",
        folder='Reconstruction',
        plot_by=['image', 'models', 'num_cells_per_dim'],
        axes_xy_proportions=(15, 15),
        difference=False,
        plot_curve=True,
        plot_curve_winner=False,
        plot_vh_classification=False,
        plot_singular_cells=False,
        plot_original_image=True,
        numbers_on=True,
        plot_again=True,
        num_cores=15,
        winner_color_dict=winner_color_dict
    )
