from operator import le, gt
from typing import Union, Tuple

import numpy as np
from matplotlib import pylab as plt

import config
from config import subcell_paper_path
from PerplexityLab.LaTexReports import Code2LatexConnector
from PerplexityLab.visualization import save_fig


def calculate_averages_from_image(image, num_cells_per_dim: Union[int, Tuple[int, int]]):
    # Example of how to calculate the averages in a single pass:
    # np.arange(6 * 10).reshape((6, 10)).reshape((2, 3, 5, 2)).mean(-1).mean(-2)
    img_x, img_y = np.shape(image)
    ncx, ncy = (num_cells_per_dim, num_cells_per_dim) if isinstance(num_cells_per_dim, int) else num_cells_per_dim
    return image.reshape((ncx, img_x // ncx, ncy, img_y // ncy)).mean(-1).mean(-2)


def load_image(image_name):
    image = plt.imread(f"{config.images_path}/{image_name}", format=image_name.split(".")[-1])
    image = np.mean(image, axis=tuple(np.arange(2, len(np.shape(image)), dtype=int)))
    image -= np.min(image)
    image /= np.max(image)
    return image


def radial_wave(image, amplitude, threshold=0.5, condition=le):
    h, w = np.shape(image)
    y, x = np.meshgrid(*list(map(range, np.shape(image))))
    d = np.sqrt((x - h / 2) ** 2 + (y - w / 2) ** 2)
    image += amplitude * (condition(image, threshold) * np.sin(2 * np.pi * d * 3 / w))
    return image


def traversal_wave(image, amplitude, threshold=0.5, condition=gt):
    h, w = np.shape(image)
    y, x = np.meshgrid(*list(map(range, np.shape(image))))
    image += amplitude * (condition(image, threshold) * np.cos(2 * np.pi * x * 5 / w))
    return image


def plot_images(image, cmap):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap)
    ax.axis('off')


if __name__ == "__main__":
    image_shape = (1680, 1680)
    families_cmaps = {
        "R": plt.get_cmap("Greys"),
        "P": plt.get_cmap("Greys"),
        "V": plt.get_cmap("Greys"),
        "F": plt.get_cmap("Greys")
    }

    report = Code2LatexConnector(path=subcell_paper_path, filename='main')

    # Regular family
    with save_fig(report.get_plot_path(section='families'), "radialwave.pdf"):
        plot_images(radial_wave(np.zeros(image_shape), amplitude=0.5), cmap=families_cmaps["R"])
    with save_fig(report.get_plot_path(section='families'), "traversalwave.pdf"):
        plot_images(traversal_wave(np.ones(image_shape), amplitude=0.5), cmap=families_cmaps["R"])

    # Piecewise constant family
    with save_fig(report.get_plot_path(section='families'), "elipse.pdf"):
        plot_images(load_image("Elipsoid_1680x1680.jpg"), cmap=families_cmaps["P"])

    # Piecewise constant family with vertices
    with save_fig(report.get_plot_path(section='families'), "shapevertex.pdf"):
        plot_images(load_image("ShapesVertex_1680x1680.jpg"), cmap=families_cmaps["V"])
    with save_fig(report.get_plot_path(section='families'), "handvertex.pdf"):
        plot_images(load_image("HandVertex_1680x1680.jpg"), cmap=families_cmaps["V"])

    # Piecewise regular family with vertices
    with save_fig(report.get_plot_path(section='families'), "regularshapevertex.pdf"):
        image = load_image("ShapesVertex_1680x1680.jpg")
        image = traversal_wave(image, amplitude=0.5) + radial_wave(image, amplitude=0.5)
        plot_images(image, cmap=families_cmaps["F"])
    with save_fig(report.get_plot_path(section='families'), "regularelipse.pdf"):
        image = load_image("Elipsoid_1680x1680.jpg")
        image = traversal_wave(image, amplitude=0.5) + radial_wave(image, amplitude=0.5)
        plot_images(image, cmap=families_cmaps["F"])
