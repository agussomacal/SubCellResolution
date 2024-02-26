import operator
import time
from functools import partial

import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial, copy_main_script_version
from PerplexityLab.visualization import generic_plot
from experiments.OtherExperiments.MLTraining.ml_global_params import num_cores
from experiments.global_params import OBERA_ITERS, \
    CCExtraWeight
from experiments.PaperPlots.exploring_methods_convergence import efficient_reconstruction, error_l1, error_linf, obtain_images, \
    obtain_image4error
from lib.AuxiliaryStructures.Constants import REGULAR_CELL, CURVE_CELL
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_regular_cells_by_minmax
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesCurveCellCreator
from lib.CellCreators.RegularCellCreator import PiecewiseConstantRegularCellCreator
from lib.CellIterators import iterate_by_condition_on_smoothness
from lib.CellOrientators import BaseOrientator, OrientByGradient
from lib.Curves.VanderCurves import CurveVandermondePolynomial
from lib.SmoothnessCalculators import naive_piece_wise
from lib.StencilCreators import StencilCreatorFixedShape, StencilCreatorAdaptive
from lib.SubCellReconstruction import SubCellReconstruction, CellCreatorPipeline, \
    ReconstructionErrorMeasure


# ========== =========== ========== =========== #
#            Experiments definition             #
# ========== =========== ========== =========== #
def get_sub_cell_model(curve_cell_creator, refinement, name, iterations, central_cell_extra_weight, metric,
                       stencil_creator=StencilCreatorFixedShape((3, 3)),
                       orientator=OrientByGradient(kernel_size=(5, 5), dimensionality=2, angle_threshold=45),
                       stencil_creator4error=None):
    return SubCellReconstruction(
        name=name,
        smoothness_calculator=naive_piece_wise,
        reconstruction_error_measure=ReconstructionErrorMeasure(stencil_creator4error,
                                                                metric=metric,
                                                                central_cell_extra_weight=central_cell_extra_weight),
        refinement=refinement,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_condition_on_smoothness, value=REGULAR_CELL,
                                      condition=operator.eq),  # only regular cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
                cell_creator=PiecewiseConstantRegularCellCreator(
                    apriori_up_value=1, apriori_down_value=0, dimensionality=2)
            ),
            # curve cell
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_condition_on_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=orientator,
                stencil_creator=stencil_creator,
                cell_creator=curve_cell_creator(regular_opposite_cell_searcher=get_opposite_regular_cells_by_minmax),
                reconstruction_error_measure=None
            )
        ],
        obera_iterations=iterations
    )


def fit_model(image, image4error, noise, sub_discretization2bound_error, metric, weight, natural_params,
              degree, obera_iterations, adaptive_stencil):
    np.random.seed(42)
    avg_values = image + np.random.uniform(-noise, noise, size=image.shape)
    print("Doing: metric ", metric, "w ", weight, "np ", natural_params, "deg ", degree, "iter ", obera_iterations)
    model = get_sub_cell_model(
        curve_cell_creator=partial(ValuesCurveCellCreator,
                                   vander_curve=partial(CurveVandermondePolynomial, degree=degree, ccew=weight),
                                   natural_params=natural_params), refinement=1,
        name="OBERA", iterations=obera_iterations, central_cell_extra_weight=CCExtraWeight, metric=metric,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                               independent_dim_stencil_size=degree + 1) if adaptive_stencil else StencilCreatorFixedShape(
            (degree + 1, degree + 1)),
        stencil_creator4error=StencilCreatorFixedShape((degree + 1, degree + 1)))

    t0 = time.time()
    model.fit(average_values=avg_values, indexer=ArrayIndexerNd(avg_values, "cyclic"))
    t_fit = time.time() - t0

    reconstruction, t_reconstruct = efficient_reconstruction(model, avg_values, sub_discretization2bound_error)

    return {
        "model": model,
        "time_to_fit": t_fit,
        "error_l1": error_l1(reconstruction, image4error),
        "error_linf": error_linf(reconstruction, image4error),
        "time_to_reconstruct": t_reconstruct
    }


if __name__ == "__main__":
    # ========== =========== ========== =========== #
    #                Experiment Run                 #
    # ========== =========== ========== =========== #
    data_manager = DataManager(
        path=config.results_path,
        name=f'OBERA_POLY_VARIANTS',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )
    # data_manager.load()

    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "precompute_images",
        obtain_images
    )

    lab.define_new_block_of_functions(
        "precompute_error_resolution",
        obtain_image4error
    )

    lab.define_new_block_of_functions(
        "models",
        fit_model,
        recalculate=False
    )
    # num_cells_per_dim = np.logspace(np.log10(20), np.log10(100), num=20, dtype=int).tolist()[:1]
    num_cells_per_dim = np.logspace(np.log10(10), np.log10(100), num=10, dtype=int).tolist()
    # num_cells_per_dim = np.logspace(np.log10(10), np.log10(20), num=5, dtype=int,
    #                                 endpoint=False).tolist() + num_cells_per_dim
    lab.execute(
        data_manager,
        num_cores=num_cores,
        forget=False,
        save_on_iteration=100,
        num_cells_per_dim=num_cells_per_dim,
        noise=[0],
        shape_name=[
            "Circle"
        ],
        sub_discretization2bound_error=[10],
        metric=[1],
        weight=[1, 100],
        natural_params=[True, False],
        degree=[2, 4],
        adaptive_stencil=[True, False],
        obera_iterations=[OBERA_ITERS]
    )

    # ========== =========== ========== =========== #
    #               Experiment Plots                #
    # ========== =========== ========== =========== #
    generic_plot(data_manager,
                 name=f"Convergence",
                 x="num_cells_per_dim", y="error_l1", label="weight",
                 num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.lineplot, palette=sns.color_palette(), marker="."),
                 log="xy",
                 plot_by=["degree"],
                 axes_by=["metric", "natural_params"],
                 format=".png",
                 xlabel=r"$1/h$",
                 ylabel=r"$||u-\tilde u ||_{L^1}$"
                 )
    generic_plot(data_manager,
                 name=f"Convergence",
                 x="num_cells_per_dim", y="error_l1", label="method",
                 num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.lineplot, palette=sns.color_palette()),
                 method=lambda metric, natural_params, adaptive_stencil: f"l{metric} params {natural_params} adapt {adaptive_stencil}",
                 log="xy",
                 plot_by=["degree"],
                 axes_by=["weight"],
                 format=".png",
                 xlabel=r"$1/h$",
                 ylabel=r"$||u-\tilde u ||_{L^1}$"
                 )

    print("CO2 consumption: ", data_manager.CO2kg)
    copy_main_script_version(__file__, data_manager.path)
