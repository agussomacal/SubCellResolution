import time
import warnings
from builtins import bool
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, Union, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special._precompute import lambertw

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from experiments.OtherExperiments.SubcellExperiments.models2compare import quadratic
from experiments.global_params import EVALUATIONS
from experiments.global_params import image_format, cred
from experiments.tools import calculate_averages_from_image, reconstruct
from experiments.tools import load_image
from experiments.tools4binary_images import fit_model, plx_plot_reconstruction4img, plot_reconstruction4img
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from perplexitylab.miscellaneous import check_do_save_or_load_experiment
from perplexitylab.plot_tools import save_figure

experiment_path = config.results_path.joinpath("Subdivision")
experiment_path.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentInfo:
    label: str
    image_name: str
    num_cells_per_dim: int
    sub_cell_model: Callable
    refinement: int = 1
    reconstruction_factor: int = 1
    angle_threshold: float = 0.
    # color: Union[str, Tuple] = "black"
    # linestyle: str = 'dashed'
    # marker: str = '.'
    recalculate: bool = False

    @property
    def identifier(self):
        return f"Img{self.image_name.split('.')[0]}_{self.num_cells_per_dim}x{self.num_cells_per_dim}_{self.label}_Ref{self.refinement}"


def single_experiment_vars_filter(experiment_info: ExperimentInfo, *args, **kwargs):
    return (experiment_info.identifier,)


def image_to_avg(experiment_info: ExperimentInfo, image, noise=0, seed=42):
    avg_values = calculate_averages_from_image(image, experiment_info.num_cells_per_dim)
    np.random.seed(seed)
    return avg_values + np.random.uniform(-noise, noise, size=avg_values.shape)


@check_do_save_or_load_experiment(default_path=experiment_path, vars_filter=single_experiment_vars_filter)
def fit_model(experiment_info: ExperimentInfo, avg_values):
    model = experiment_info.sub_cell_model(refinement=experiment_info.refinement,
                                           angle_threshold=experiment_info.angle_threshold)

    t0 = time.time()
    model.fit(average_values=avg_values, indexer=ArrayIndexerNd(avg_values, "cyclic"))
    t_fit = time.time() - t0
    print("Time to fit model:", t_fit)
    return model


@check_do_save_or_load_experiment(default_path=experiment_path, vars_filter=single_experiment_vars_filter,
                                  saver=lambda data, filepath: plt.imsave(filepath, data),
                                  loader=lambda filepath: load_image(filepath, other_path=""),
                                  file_format="png")
def do_reconstruction(experiment_info: ExperimentInfo, image, model):
    t0 = time.time()
    reconstruction = reconstruct(image, model.cells, model.resolution, experiment_info.reconstruction_factor,
                                 do_evaluations=EVALUATIONS)
    t_reconstruct = time.time() - t0
    print("Time to reconstruct:", t_reconstruct)
    return reconstruction


if __name__ == "__main__":

    # Experiment general params
    noise = 0
    seed = 42
    recalculate_all = True

    # Reconstruction plot params
    matplotlib.rcParams['text.usetex'] = False
    curve_color = cred
    cmap_reconstruction = "Reds"
    cmap_true_image = "Greys_r"
    fig_size = (15, 15)

    # ---------- Experiment list ---------- #
    experiment_list = \
        [
            ExperimentInfo(
                label="AEROS quadratic",
                image_name="batata.jpg",
                num_cells_per_dim=20,
                sub_cell_model=quadratic,
                refinement=1,
                recalculate=False,
            ),
            ExperimentInfo(
                label="AEROS quadratic",
                image_name="batata.jpg",
                num_cells_per_dim=20,
                sub_cell_model=quadratic,
                refinement=2,
                recalculate=False,
            )
        ]

    # ---------- Do experiments ---------- #
    for experiment_info in experiment_list:
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
