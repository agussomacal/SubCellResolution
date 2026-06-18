import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from experiments.Refinement.ex_refinement_config import experiment_path
from experiments.global_params import EVALUATIONS
from experiments.tools import calculate_averages_from_image, load_image, reconstruct
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from perplexitylab.experiment_tools import perplexifier


def single_experiment_vars_filter(*args, **kwargs):
    return list(map(str, args)) + list(map(str, kwargs.values()))


def image_to_avg(num_cells_per_dim, image, noise=0, seed=42):
    avg_values = calculate_averages_from_image(image, num_cells_per_dim)
    np.random.seed(seed)
    return avg_values + np.random.uniform(-noise, noise, size=avg_values.shape)


@perplexifier(default_path=experiment_path)
def fit_model(sub_cell_model, angle_threshold, refinement, avg_values):
    model = sub_cell_model(refinement=refinement, angle_threshold=angle_threshold)

    t0 = time.time()
    model.fit(average_values=avg_values, indexer=ArrayIndexerNd(avg_values, "cyclic"))
    t_fit = time.time() - t0
    print("\n\tTime to fit model:", t_fit)
    return model


@perplexifier(default_path=experiment_path,
              saver=lambda data, filepath: plt.imsave(filepath, data),
              loader=lambda filepath: load_image(filepath, other_path=""),
              file_format="png")
def do_reconstruction(image, model, reconstruction_factor):
    t0 = time.time()
    reconstruction = reconstruct(image, model.cells, model.resolution, reconstruction_factor,
                                 do_evaluations=EVALUATIONS)
    t_reconstruct = time.time() - t0
    print("\n\tTime to reconstruct:", t_reconstruct)
    return reconstruction


def calculate_error(image, reconstruction, p=2):
    return np.power(np.mean(np.abs(image - reconstruction) ** p), 1 / p)
