import operator
import time
from functools import partial

import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import generic_plot, one_line_iterator, perplex_plot
from experiments.VizReconstructionUtils import plot_cells, draw_cell_borders, plot_cells_identity, \
    plot_cells_vh_classification_core, plot_cells_not_regular_classification_core, plot_curve_core
from experiments.subcell_paper.global_params import CurveAverageQuadraticCC, CCExtraWeight, cpink, corange, cyellow, \
    cblue, cgreen, runsinfo, cbrown, cgray, cpurple, cred, ccyan, EVALUATIONS
from experiments.subcell_paper.ex_scheme import model_color, names_dict
from experiments.subcell_paper.models2compare import upwind, elvira, aero_linear, quadratic, aero_lq_vertex, \
    scheme_error, scheme_reconstruction_error
from experiments.subcell_paper.tools import get_reconstruction_error, calculate_averages_from_image, load_image, \
    reconstruct
from lib.AuxiliaryStructures.Constants import REGULAR_CELL, CURVE_CELL
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import REGULAR_CELL_TYPE
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_regular_cells, \
    get_opposite_regular_cells_by_stencil
from lib.CellCreators.CurveCellCreators.TaylorCurveCellCreator import TaylorCircleCurveCellCreator
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesCurveCellCreator, \
    ValuesLineConsistentCurveCellCreator
from lib.CellCreators.CurveCellCreators.VertexCellCreator import LinearVertexCellCurveCellCreator
from lib.CellCreators.RegularCellCreator import PiecewiseConstantRegularCellCreator, MirrorCellCreator
from lib.CellCreators.VertexCellCreators.VertexCellCreatorBase import VertexCellCreatorUsingNeighboursLines
from lib.CellIterators import iterate_by_reconstruction_error_and_smoothness, \
    iterate_all
from lib.CellOrientators import BaseOrientator, OrientByGradient, OrientPredefined
from lib.SmoothnessCalculators import naive_piece_wise
from lib.StencilCreators import StencilCreatorAdaptive, StencilCreatorFixedShape
from lib.SubCellReconstruction import SubCellReconstruction, ReconstructionErrorMeasure, CellCreatorPipeline, \
    keep_cells_on_condition, curve_condition, ReconstructionErrorMeasureDefaultStencil
from lib.SubCellScheme import SubCellScheme


# ========== ========== Experiment definitions ========== ========== #
def calculate_true_solution(image, num_cells_per_dim, velocity, ntimes):
    image = load_image(image)
    pixels_per_cell = np.array(np.shape(image)) / num_cells_per_dim
    velocity_in_pixels = np.array(pixels_per_cell * np.array(velocity), dtype=int)
    assert np.all(velocity_in_pixels == pixels_per_cell * np.array(velocity))

    true_solution = []
    true_reconstruction = []
    for i in range(ntimes + 1):
        true_reconstruction.append(image.copy())
        true_solution.append(calculate_averages_from_image(image, num_cells_per_dim))
        image = np.roll(image, velocity_in_pixels)
    return {
        "true_solution": true_solution,
        "true_reconstruction": np.array(true_reconstruction)[[0, -2]]
    }


def fit_model(subcell_reconstruction):
    def decorated_func(image, noise, num_cells_per_dim, reconstruction_factor, velocity, ntimes, true_solution):
        image_array = load_image(image)
        avg_values = calculate_averages_from_image(image_array, num_cells_per_dim)
        np.random.seed(42)
        avg_values = avg_values + np.random.uniform(-noise, noise, size=avg_values.shape)

        model = SubCellScheme(name=subcell_reconstruction.__name__, subcell_reconstructor=subcell_reconstruction(),
                              min_value=0, max_value=1)

        t0 = time.time()
        solution, all_cells = model.evolve(
            init_average_values=avg_values, indexer=ArrayIndexerNd(avg_values, "cyclic"),
            velocity=np.array(velocity), ntimes=ntimes,
            # interface_oracle=None
            interface_oracle=(np.array(true_solution) > 0) * (np.array(true_solution) < 1)
        )
        t_fit = time.time() - t0

        # reconstruct at the begining and at the end
        t0 = time.time()
        reconstruction = [
            reconstruct(image_array, all_cells[0], model.resolution, reconstruction_factor,
                        do_evaluations=EVALUATIONS),
            reconstruct(image_array, all_cells[-1], model.resolution, reconstruction_factor,
                        do_evaluations=EVALUATIONS)
        ]
        t_reconstruct = time.time() - t0

        return {
            "model": model,
            "time_to_fit": t_fit,
            "reconstruction": reconstruction,
            "cells": [all_cells[0], all_cells[-1]],
            "solution": solution,
            "time_to_reconstruct": t_reconstruct
        }

    # need to change the name so the lab experiment saves the correct name and not the uniformly "decorated_func"
    # the other option is to pass to the block the name we wish to associate to the function.
    decorated_func.__name__ = subcell_reconstruction.__name__
    return decorated_func


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='SchemesConvergence',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )

    lab = LabPipeline()

    lab.define_new_block_of_functions(
        "ground_truth",
        calculate_true_solution,
        recalculate=False
    )

    lab.define_new_block_of_functions(
        "models",
        *map(fit_model, [
            upwind,
            elvira,
            aero_linear,
            quadratic,
            aero_lq_vertex
        ]),
        recalculate=False
    )

    ntimes = 20
    lab.execute(
        data_manager,
        num_cores=15,
        forget=False,
        save_on_iteration=None,
        refinement=[1],
        ntimes=[ntimes],
        velocity=[(0, 1 / 4)],
        num_cells_per_dim=[10, 15, 20, 28, 30, 42],  # 60 10, 15,
        noise=[0],
        image=[
            # "yoda.jpg",
            # "DarthVader.jpeg",
            # "Ellipsoid_1680x1680.jpg",
            "ShapesVertex_1680x1680.jpg",
            # "HandVertex_1680x1680.jpg",
            # "Polygon_1680x1680.jpg",
        ],
        iterations=[0],  # 500
        reconstruction_factor=[1],
    )

    generic_plot(data_manager,
                 name="ErrorConvergence",
                 x="N",
                 y="final_error",
                 label="method",
                 plot_by=["image"],
                 N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 final_error=lambda image, true_solution, solution: scheme_error(image, true_solution, solution)[-1],
                 plot_func=NamedPartial(
                     sns.lineplot, marker="o", linestyle="--",
                     palette={v: model_color[k] for k, v in names_dict.items()}
                 ),
                 models=list(model_color.keys()),
                 method=lambda models: names_dict[models],
                 log="y",
                 )

    generic_plot(data_manager,
                 name="ReconstructionErrorConvergence",
                 x="N",
                 y="final_error",
                 label="method",
                 plot_by=["image"],
                 N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 times=lambda ntimes: np.arange(ntimes),
                 final_error=lambda true_reconstruction, reconstruction, reconstruction_factor:
                 scheme_reconstruction_error(true_reconstruction, reconstruction, reconstruction_factor)[-1],
                 plot_func=NamedPartial(
                     sns.lineplot, marker="o", linestyle="--",
                     palette={v: model_color[k] for k, v in names_dict.items()}
                 ),
                 log="y",
                 models=list(model_color.keys()),
                 method=lambda models: names_dict[models],
                 )

    generic_plot(data_manager,
                 name="RelativeErrorConvergence",
                 x="N",
                 y="final_error",
                 label="method",
                 plot_by=["image"],
                 N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 final_error=lambda image, true_solution, solution: np.divide(
                     *scheme_error(image, true_solution, solution)[[0, -1]][::-1]),
                 plot_func=NamedPartial(
                     sns.lineplot, marker="o", linestyle="--",
                     palette={v: model_color[k] for k, v in names_dict.items()}
                 ),
                 models=list(model_color.keys()),
                 method=lambda models: names_dict[models],
                 log="y",
                 )

    generic_plot(data_manager,
                 name="RelativeReconstructionErrorConvergence",
                 x="N",
                 y="final_error",
                 label="method",
                 plot_by=["image"],
                 N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 times=lambda ntimes: np.arange(ntimes),
                 final_error=lambda true_reconstruction, reconstruction, reconstruction_factor:
                 np.divide(
                     *scheme_reconstruction_error(true_reconstruction, reconstruction, reconstruction_factor)[[0, -1]][
                      ::-1]),
                 plot_func=NamedPartial(
                     sns.lineplot, marker="o", linestyle="--",
                     palette={v: model_color[k] for k, v in names_dict.items()}
                 ),
                 log="y",
                 models=list(model_color.keys()),
                 method=lambda models: names_dict[models],
                 )
    # generic_plot(data_manager,
    #              name="Time2Fit",
    #              x="image",
    #              y="time_to_fit",
    #              label="method",
    #              plot_by=["num_cells_per_dim", "image"],
    #              # models=["elvira", "quadratic"],
    #              times=lambda ntimes: np.arange(ntimes),
    #              plot_func=NamedPartial(
    #                  sns.barplot,
    #                  palette={v: model_color[k] for k, v in names_dict.items()}
    #              ),
    #              log="y",
    #              models=list(model_color.keys()),
    #              method=lambda models: names_dict[models],
    #              )
