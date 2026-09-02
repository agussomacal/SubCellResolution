import itertools

from experiments.Refinement.ex_refinement_config import experiment_subdivision_path
from experiments.tools import calculate_averages_from_curve
from lib.Curves.CurveCircle import CurveCircle, CircleParams
from lib.Curves.CurveTrigo import CurveTrigo, TrigoParams
from lib.Curves.Curves import Curve
from perplexitylab.plot_tools import save_figure

# --------- PATHs to experiment --------- #
experiment_path = experiment_subdivision_path.joinpath("TestCases")
experiment_path.mkdir(parents=True, exist_ok=True)

# --------- Experiment parameters --------- #
shape_names_for_label_plots = {
    "Circle": "Circle",
    "Sinusoidal-horizon": "Sinusoidal-horizon",
}

shapes = {
    "Circle": CurveCircle(params=CircleParams(x0=0.511, y0=0.486, radius=0.232)),
    "Sinusoidal-horizon": CurveTrigo(params=TrigoParams(x0=0.5, y0=0.5, amplitude=0.1, frequency=1.))
}

num_cells_per_dim = {
    20: 20,
    40: 40,
    "HD": 500
}


def get_shape_key(shape: Curve):
    return list(shapes.keys())[list(shapes.values()).index(shape)]


if __name__ == "__main__":

    # --------- Plot parameters --------- #
    cmap = "viridis"
    vmax = 1
    vmin = -1
    alpha = 1

    # --------- Do the plots --------- #
    for shape, n in itertools.product(shapes, num_cells_per_dim):
        avg_values = calculate_averages_from_curve(shapes[shape], (num_cells_per_dim[n], num_cells_per_dim[n]))

        with save_figure(filename=f"{shape}_{n}", path=experiment_path, figsize=(8, 8), show=False) as (fig, ax):
            ax.imshow(avg_values, cmap=cmap, vmax=vmax, vmin=vmin, alpha=alpha)
            ax.minorticks_off()
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                left=False,  # ticks along the bottom edge are off
                right=False,  # ticks along the top edge are off
                labelbottom=False,
                labeltop=False,
                labelleft=False,
                labelright=False
            )  # labels along the bottom edge are off
