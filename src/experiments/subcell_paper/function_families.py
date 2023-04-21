from operator import le, gt

import numpy as np
from matplotlib import pylab as plt

import config
from config import subcell_paper_path
from src.LaTexReports import Code2LatexConnector
from src.viz_utils import save_fig


def load_image(image_name):
    image = plt.imread(f"{config.images_path}/{image_name}", format=image_name.split(".")[-1])
    image = np.mean(image, axis=tuple(np.arange(2, len(np.shape(image)), dtype=int)))
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
