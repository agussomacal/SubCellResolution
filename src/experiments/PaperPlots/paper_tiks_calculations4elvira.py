import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.polynomial.polynomial import Polynomial

from experiments.VizReconstructionUtils import plot_cells
from experiments.tools import get_evaluations2test_curve
from lib.Curves.CurvePolynomial import CurvePolynomial
from lib.DataManagers.DatasetsManagers.DatasetsBaseManager import get_averages_from_curve_kernel

kernel_size = (3, 3)
poly = 0.5 * Polynomial([0, 1]) + 1.35 + 0.25 * Polynomial([0, 1]) ** 2 - 1.5
value_up = 0
value_down = 1
refinement = 20

curve = CurvePolynomial(poly, value_up, value_down)
kernel = get_averages_from_curve_kernel(kernel_size, curve, center_cell_coords=None)
u = get_evaluations2test_curve(curve, kernel_size, refinement=refinement)

kernel_vertical = kernel.sum(axis=1)

Polynomial([kernel_vertical[1], kernel_vertical[1] - kernel_vertical[0]])
Polynomial([kernel_vertical[1], kernel_vertical[2] - kernel_vertical[0]])
Polynomial([kernel_vertical[1], kernel_vertical[2] - kernel_vertical[1]])

fig = plt.figure()
ax = fig.add_gridspec(6, 5)
ax1 = fig.add_subplot(ax[:, 0:3])
ax1.set_title('Averages')
ax2 = fig.add_subplot(ax[:, 3:])
ax2.set_title('True curve')

sns.heatmap(kernel, annot=True, cmap="viridis", alpha=0.7, ax=ax1)
plot_cells(ax=ax2, colors=u, mesh_shape=np.array(kernel_size) * refinement, alpha=0.5,
           cmap="viridis",
           vmin=-1, vmax=1)
plt.tight_layout()
plt.show()
