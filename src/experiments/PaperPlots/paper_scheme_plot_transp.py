import operator
import warnings
from functools import partial

import matplotlib.colors as mcolors
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial, copy_main_script_version
from PerplexityLab.visualization import generic_plot, LegendOutsidePlot, perplex_plot, \
    one_line_iterator
from experiments.PaperPlots.exploring_scheme_methods import scheme_reconstruction_error, \
    calculate_true_solution, fit_model
from experiments.PaperPlots.models2compare import piecewise
from experiments.PaperPlots.paper_corners_plot import elvira_cc, \
    reconstruction_error_measure_default, piecewise01, aero_q, tem, reconstruction_error_measure_w
from experiments.VizReconstructionUtils import plot_cells, plot_curve_core, draw_numbers
from experiments.global_params import cpink, corange, cblue, cgreen, runsinfo, cpurple, cred, cgray, \
    RESOLUTION_FACTOR, num_cores, running_in, image_format
from lib.AuxiliaryStructures.Constants import CURVE_CELL
from lib.CellClassifiers import cell_classifier_try_it_all
from lib.CellCreators.CellCreatorBase import CURVE_CELL_TYPE, VERTEX_CELL_TYPE
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_regular_cells_by_minmax
from lib.CellCreators.CurveCellCreators.VertexCellCreator import LinearVertexCellCurveCellCreator
from lib.CellIterators import iterate_by_reconstruction_error_and_smoothness
from lib.CellOrientators import OrientByGradient
from lib.SmoothnessCalculators import naive_piece_wise, indifferent
from lib.StencilCreators import StencilCreatorAdaptive
from lib.SubCellReconstruction import SubCellReconstruction, CellCreatorPipeline

num_screenshots = 20

num_cells_per_dim = [15, 30, 60]
num_cells_per_dim = [30, 60]


def zalesak_notched_circle(num_pixels=1680):
    radius = num_pixels / 3
    rectangle_width = radius / 3  # / 3
    rectangle_y_shift = rectangle_width
    rectangle_height = rectangle_width * 4
    center = num_pixels // 2 * np.ones(2)
    image = np.zeros((num_pixels, num_pixels))
    xycoords = np.array(np.meshgrid(range(num_pixels), range(num_pixels)))
    image[np.where(np.sum((xycoords - center[:, np.newaxis, np.newaxis]) ** 2, axis=0) <= radius ** 2)] = 1
    image[
        np.all(xycoords > (center -
                           np.array([rectangle_width // 2, rectangle_height // 2 - rectangle_y_shift]))[:, np.newaxis,
                          np.newaxis],
               axis=0) &
        np.all(xycoords < (center +
                           np.array([rectangle_width // 2, rectangle_height // 2 + rectangle_y_shift]))[:, np.newaxis,
                          np.newaxis],
               axis=0)] = 0

    return image


plt.imsave(f"{config.images_path}/zalesak_notched_circle.jpg", zalesak_notched_circle(num_pixels=1680), cmap='gray')

# ========== ========== Names and colors to present ========== ========== #
names_dict = {
    "aero_qelvira_vertex": "ELVIRA-WO + AEROS Quadratic + TEM + AEROS Vertex",

    "elvira": "ELVIRA",
    "elvira_w_oriented": "ELVIRA-WO",

    "linear_obera": "OBERA Linear",
    "linear_obera_w": "OBERA-W Linear",

    "quadratic_obera_non_adaptive": "OBERA Quadratic",
    "quadratic_aero": "AEROS Quadratic",

    "upwind": "UpWind",
}
model_color = {
    "aero_qelvira_vertex": cblue,

    "elvira": corange,
    "elvira_w_oriented": cred,

    "linear_obera": cpink,
    "linear_obera_w": cpurple,

    "quadratic_obera_non_adaptive": cgreen,
    "quadratic_aero": cgreen,

    "upwind": cgray,
}
names_dict = {k: names_dict[k] for k in model_color.keys()}

runsinfo.append_info(
    **{k.replace("_", "-"): v for k, v in names_dict.items()}
)


@perplex_plot(legend=False)
@one_line_iterator()
def plot_reconstruction_init_final(fig, ax, true_reconstruction, num_cells_per_dim, resolution, cells,
                                   alpha_true_image=0.5, numbers_on=False, default_linewidth=3.5,
                                   cmap_true_image="Greys_r", init_end_color=(cgreen, cblue),
                                   trim=((0, 1), (0, 1)), vmin=None, vmax=None, labels=True):
    if isinstance(init_end_color[0], str):
        init_end_color = tuple(list(map(mcolors.to_rgb, init_end_color)))

    model_resolution = np.array(resolution)
    image = true_reconstruction[0]

    if alpha_true_image > 0:
        plot_cells(ax, colors=image, mesh_shape=model_resolution, alpha=alpha_true_image, cmap=cmap_true_image,
                   vmin=np.min(image) if vmin is None else vmin,
                   vmax=np.max(image) if vmax is None else vmax,
                   labels=labels)

    for i, color in zip([0, -1], init_end_color):
        plot_curve_core(ax, curve_cells=[cell for cell in cells[i].values() if
                                         cell.CELL_TYPE in [CURVE_CELL_TYPE, VERTEX_CELL_TYPE]],
                        default_linewidth=default_linewidth * 20 / num_cells_per_dim,
                        color=color)

    ax.set_ylim((model_resolution[1] - trim[0][1] - 0.5, -0.5 + trim[0][0]))
    ax.set_xlim((trim[1][0] - 0.5, model_resolution[0] - trim[1][1] - 0.5))

    draw_numbers(
        ax, mesh_shape=num_cells_per_dim,
        refinement=model_resolution // num_cells_per_dim,
        numbers_on=numbers_on,
        prop_ticks=10 / num_cells_per_dim  # each 10 cells a tick
    )

    if not numbers_on:
        plt.box(False)


def elvira_w_oriented(smoothness_calculator=naive_piece_wise, refinement=1, obera_iterations=0, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            elvira_cc(angle_threshold=45),
        ],
        obera_iterations=obera_iterations
    )


def quadratic_aero(smoothness_calculator=naive_piece_wise, refinement=1, obera_iterations=0, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            aero_q(angle_threshold=45),
        ],
        obera_iterations=obera_iterations,
    )


def aero_qelvira_vertex(smoothness_calculator=naive_piece_wise, refinement=1, obera_iterations=0, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise01,
            elvira_cc(angle_threshold=45),
            aero_q(angle_threshold=45),
            tem(reconstruction_error_measure_w),
            # ------------ AVRO ------------ #
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2, method="sobel",
                                            angle_threshold=0),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=4),
                cell_creator=LinearVertexCellCurveCellCreator(
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_minmax),
                reconstruction_error_measure=reconstruction_error_measure_w
            ),
        ],
        obera_iterations=obera_iterations,
        eps_complexity=[0, 0, 1e-3, 0, 0]
    )


def upwind(smoothness_calculator=indifferent, refinement=1, *args, **kwargs):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        cell_classifier=cell_classifier_try_it_all,
        reconstruction_error_measure=reconstruction_error_measure_default,
        refinement=refinement,
        cell_creators=
        [
            piecewise
        ],
        obera_iterations=0
    )


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.paper_results_path,
        emissions_path=config.results_path,
        name=f'SchemesRev',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )

    lab = LabPipeline()

    lab.define_new_block_of_functions(
        "ntimescalcul",
        lambda num_cells_per_dim, velocity: {
            "ntimes": int(np.round(num_cells_per_dim / velocity[1])) + 1},
        recalculate=False
    )

    lab.define_new_block_of_functions(
        "ground_truth",
        calculate_true_solution,
        recalculate=False
    )

    lab.define_new_block_of_functions(
        "models",
        *map(fit_model, [
            aero_qelvira_vertex,
            # elvira,
            elvira_w_oriented,
            quadratic_aero,
            upwind,
        ]),
        recalculate=False
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lab.execute(
            data_manager,
            num_cores=num_cores,
            forget=False,
            save_on_iteration=None,
            refinement=[1],
            # velocity=[(0, 1), (0, 0.5), (0, 0.25)],
            velocity=[(0, 0.25)],
            angular_velocity=[0],
            num_cells_per_dim=num_cells_per_dim,  # 60
            noise=[0],
            image=[
                "batata.jpg",
                "ShapesVertex.jpg",
                "zalesak_notched_circle.jpg",
            ],
            reconstruction_factor=[RESOLUTION_FACTOR],
            num_screenshots=[num_screenshots],
        )
    print(set(data_manager["models"]))

    # ---------------- Paper plots ---------------- #
    for ncpd, legend in zip([30, 60], [False, True]):
        generic_plot(data_manager,
                     name="ReconstructionErrorInTimex",
                     format=image_format,
                     path=config.subcell_paper_figures_path,
                     x="times", y="scheme_error", label="method",
                     # axes_by=["num_cells_per_dim"],
                     num_cells_per_dim=ncpd,
                     add_legend=legend,
                     plot_by=["num_cells_per_dim", "image", "angular_velocity", "velocity"],
                     times=lambda ntimes, num_screenshots: np.linspace(0, ntimes, num_screenshots, dtype=int),
                     scheme_error=scheme_reconstruction_error,
                     plot_func=NamedPartial(
                         sns.lineplot,
                         marker="o", linestyle="--",
                         markers=True, linewidth=3,
                         palette={v: model_color[k] for k, v in names_dict.items()}
                     ),
                     axes_xy_proportions=(12, (5 if legend else 4.2)), transpose=True,
                     share_axis="x", log="y",
                     models=list(model_color.keys()),
                     method=lambda models: names_dict[models],
                     # num_cells_per_dim=[30],
                     axis_font_dict={'color': 'black', 'weight': 'normal', 'size': 25},
                     labels_font_dict={'color': 'black', 'weight': 'normal', 'size': 25},
                     legend_font_dict={'weight': 'normal', "size": 19, 'stretch': 'normal'},
                     uselatex=False if running_in == "server" else True,
                     xlabel=r"Iterations",
                     ylabel=r"$\|u-\tilde u \|_{L^1}$",
                     # xticks=[1, 2, 4, 8, 16, 24, 40, 70, 120],
                     create_preimage_data=True if running_in == "server" else False,
                     use_preimage_data=True,
                     only_create_preimage_data=True if running_in == "server" else False,
                     legend_outside_plot=LegendOutsidePlot(loc="lower center",
                                                           extra_y_top=0.01,
                                                           extra_y_bottom=0.15 + (0.25 if legend else 0),
                                                           extra_x_left=0.125, extra_x_right=0.075),
                     )

    for i, ncpd in enumerate(num_cells_per_dim):
        plot_reconstruction_init_final(
            data_manager,
            path=config.subcell_paper_figures_path,
            format=".pdf",
            plot_by=["num_cells_per_dim", "angular_velocity", "image", "velocity", "models"],
            num_cells_per_dim=ncpd,
            folder="InitEndComparison",
            alpha_true_image=0.1,
            cmap_true_image="Greys_r", init_end_color=("black", cred),
            image=[
                "batata.jpg",
                "ShapesVertex.jpg",
            ],
            trim=((2*(i+1), 3*(i+1)), (2*(i+1), 3*(i+1))), vmin=None, vmax=None, labels=False)

    for i, ncpd in enumerate(num_cells_per_dim):
        plot_reconstruction_init_final(
            data_manager,
            path=config.subcell_paper_figures_path,
            format=".pdf",
            plot_by=["num_cells_per_dim", "angular_velocity", "image", "velocity", "models"],
            folder="InitEndComparison",
            alpha_true_image=0.1,
            cmap_true_image="Greys", init_end_color=("black", cred),
            image="zalesak_notched_circle.jpg",
            trim=((2*(i+1), 2*(i+1)), (2*(i+1), 2*(i+1))), vmin=None, vmax=None, labels=False)


    print("CO2 consumption: ", data_manager.CO2kg)
    copy_main_script_version(__file__, data_manager.path)
