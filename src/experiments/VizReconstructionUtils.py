from collections import defaultdict
from functools import partial

import seaborn as sns

from lib.CellCreators.CellCreatorBase import SPECIAL_CELLS_COLOR_DICT, REGULAR_CELL_TYPE
from lib.CellCreators.CurveCellCreators.CurveCellCreatorBase import CellCurveBase
from lib.Curves.CurveVertex import CurveVertexPolynomial

# from lib.CellCreators.VertexCellCreators.VertexCellCreatorBase import VertexLinearExtended

CURVE_PLOT_RESOLUTION = 25
SUB_DISCRETIZATION2BOUND_ERROR = 5

# matplotlib.use('Agg')
from collections import namedtuple
from typing import List, Tuple

import numpy as np
from matplotlib import patches

from lib.AuxiliaryStructures.Constants import VERTICAL, HORIZONTAL

COLOR_WHITE = (1, 1, 1)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (161 / 255, 0, 0)
COLOR_BLUE = (0, 0, 1)
COLOR_GREEN = (0, 1, 0)
COLOR_YELLOW = (225 / 255, 173 / 255, 1 / 255)

VerticalCells_COLOR = COLOR_RED
HorizontalCells_COLOR = COLOR_BLUE
VHM_ORIENTATION_COLOR_DICT = {
    VERTICAL: VerticalCells_COLOR,
    HORIZONTAL: HorizontalCells_COLOR
}

COLOR_CURVE = COLOR_BLACK

SpecialCellsPlotTuple = namedtuple('SpecialCellsPlotTuple', 'name, indexes, color, alpha')


# ========= ========= ========= ========= ========= #
# ---------- Visualization auxiliary functions ---------
# ========= ========= ========= ========= ========= #
def transform_points2plot(points):
    return (np.array(points) - 0.5)[:, [1, 0]]


def calculate_proportional_linewith(mesh_shape):
    Nx, Ny = mesh_shape
    linewidth = 5 / np.sqrt(Nx * Ny)
    return linewidth


def get_ticks_positions(N, refinement, prop_ticks):
    return np.array(
        np.linspace(0, N * refinement, num=int(N * prop_ticks), endpoint=False, dtype=int) + 0.5 * refinement,
        dtype=int)


def draw_cell_borders(ax, mesh_shape, color='black', refinement=1, prop_ticks=1, numbers_on=True):
    # TODO: axis show 1 2 3 not 1.5 etc
    mesh_shape = (mesh_shape, mesh_shape) if isinstance(mesh_shape, int) else mesh_shape
    refinement = (refinement, refinement) if isinstance(refinement, int) else refinement
    Nx, Ny = mesh_shape
    linewidth = calculate_proportional_linewith(mesh_shape)
    ax.hlines(y=np.arange(Ny * refinement[1], step=refinement[1]) - 0.5, xmin=-0.5, xmax=Nx * refinement[0],
              colors=color, linestyles='solid', linewidth=linewidth)
    ax.vlines(x=np.arange(Nx * refinement[0], step=refinement[0]) - 0.5, ymin=-0.5, ymax=Ny * refinement[1],
              colors=color, linestyles='solid', linewidth=linewidth)

    ax.minorticks_off()
    if numbers_on:
        x_ticks = get_ticks_positions(Nx, refinement[0], prop_ticks)
        y_ticks = get_ticks_positions(Ny, refinement[1], prop_ticks)

        ax.xaxis.set_ticks(ticks=x_ticks, labels=x_ticks // refinement[0])
        ax.yaxis.set_ticks(ticks=y_ticks, labels=y_ticks // refinement[1])
        ax.tick_params(direction='inout', length=6, width=1, colors='k',
                       grid_color='k', grid_alpha=0.5, which='minor')

    else:
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelbottom=False, labeltop=False, labelleft=False, labelright=False
        )  # labels along the bottom edge are off


def plot_cells(ax, colors, mesh_shape=None, cmap=None, alpha=None, vmin=None, vmax=None):
    extent = np.array([0, mesh_shape[0], mesh_shape[1], 0]) - 0.5 if mesh_shape is not None else mesh_shape
    # extent = np.array([0, mesh_shape[1], mesh_shape[0], 0]) - 0.5 if mesh_shape is not None else mesh_shape
    ax.imshow(colors, interpolation=None, origin='upper', extent=extent, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()


def plot_specific_cells(ax, mesh_shape: Tuple[int], special_cells: List[SpecialCellsPlotTuple], rectangle_mode=False):
    """

    :param ax:
    :param avg_values:
    :param special_cells: [('singular', [(0,0), (0,1), ...], 'Reds'), ('regular', [(0,2), (0,3), ...], 'Blues')]
    :return:
    """
    # ----- plot special cells -----
    linewidth = calculate_proportional_linewith(mesh_shape) * 20
    for cells_name, cells_indexes, color, alpha in special_cells:
        for coords in cells_indexes:
            rect = patches.Rectangle(np.array(coords)[::-1] - 0.5, 1, 1, angle=0, linewidth=linewidth,
                                     edgecolor=color, facecolor="none" if rectangle_mode else color, alpha=alpha)
            ax.add_patch(rect)
    # if rectangle_mode:
    #
    # else:
    #     all_cells = np.zeros(np.append(mesh_shape, 4))
    #     for cells_name, cells_indexes, color, alpha in special_cells:
    #         on_color = colors.to_rgba(color, alpha)
    #         for s in cells_indexes:
    #             all_cells[s] = on_color
    #         add_legend_to_imshow(ax, color=on_color, label=cells_name)
    #     plot_cells(ax, colors=all_cells)


def is_in_square(cell_coords, square):
    cell_coords = np.reshape(cell_coords, (-1, np.shape(square)[1]))
    return np.prod((np.min(square, axis=0) <= cell_coords) * (np.max(square, axis=0) >= cell_coords), axis=1,
                   dtype=bool)


def plot_cells_not_regular_classification_core(ax, mesh_shape, all_cells, alpha=1.0):
    cell_classes = defaultdict(list)
    for coords, cell in all_cells.items():
        if cell.CELL_TYPE != REGULAR_CELL_TYPE:
            cell_classes[cell.CELL_TYPE].append(coords)

    plot_specific_cells(
        ax=ax,
        mesh_shape=mesh_shape,
        special_cells=[SpecialCellsPlotTuple(name=cell_type, indexes=coords_list,
                                             color=SPECIAL_CELLS_COLOR_DICT[cell_type],
                                             alpha=alpha) for cell_type, coords_list in cell_classes.items()],
        rectangle_mode=True)


def plot_cells_identity(ax, mesh_shape, all_cells, alpha=1.0):
    cell_types = defaultdict(list)
    for cell in all_cells.values():
        cell_types[str(cell)].append(cell.coords.tuple)

    plot_specific_cells(
        ax=ax,
        mesh_shape=mesh_shape,
        special_cells=[
            SpecialCellsPlotTuple(name=k, indexes=v, color=sns.color_palette("colorblind")[i % 8], alpha=alpha) for
            i, (k, v) in enumerate(cell_types.items())],
        rectangle_mode=True
    )


def plot_cells_type_of_curve_core(ax, mesh_shape, all_cells, alpha=1.0):
    cell_types = defaultdict(list)
    for cell in all_cells.values():
        if isinstance(cell, CellCurveBase):
            cell_types[type(cell.curve)].append(cell.coords.tuple)

    plot_specific_cells(
        ax=ax,
        mesh_shape=mesh_shape,
        special_cells=[
            SpecialCellsPlotTuple(name=k, indexes=v, color=sns.color_palette("colorblind")[i % 8], alpha=alpha) for
            i, (k, v) in enumerate(cell_types.items())],
        rectangle_mode=True
    )


def plot_cells_vh_classification_core(ax, mesh_shape, all_cells, alpha=1.0):
    vertical_cells = [cell.coords.tuple for cell in all_cells.values()
                      if hasattr(cell, "dependent_axis") and cell.dependent_axis == VERTICAL]
    horizontal_cells = [cell.coords.tuple for cell in all_cells.values()
                        if hasattr(cell, "dependent_axis") and cell.dependent_axis == HORIZONTAL]

    plot_specific_cells(
        ax=ax,
        mesh_shape=mesh_shape,
        special_cells=[
            SpecialCellsPlotTuple(name='VerticalCells', indexes=vertical_cells,
                                  color=VerticalCells_COLOR, alpha=alpha),
            SpecialCellsPlotTuple(name='HorizontalCells', indexes=horizontal_cells,
                                  color=HorizontalCells_COLOR, alpha=alpha)
        ],
        rectangle_mode=True)


def get_curve(curve_cell: CellCurveBase, coords2=None):
    x = curve_cell.coords.coords[curve_cell.independent_axis] + \
        np.linspace(0, 1, CURVE_PLOT_RESOLUTION + 1).reshape((-1, 1))
    c = curve_cell.curve(x)
    c = c.T if len(np.shape(c)) == 2 else [c]
    for y in c:
        points = np.concatenate((x, np.reshape(y, (-1, 1))), axis=1)
        points = points[:, [curve_cell.independent_axis, curve_cell.dependent_axis]]  # correct order x, y.

        square = [curve_cell.coords.coords, coords2 if coords2 is not None else curve_cell.coords.coords + 1]
        points_inside_cell = list(map(partial(is_in_square, square=square), points))
        yield points[np.ravel(points_inside_cell), :]


def get_curve_vertex(curve_cell: CurveVertexPolynomial, coords2=None):
    for point in map(np.array, [curve_cell.curve.point1, curve_cell.curve.point2]):
        points = point + (np.array(curve_cell.curve.vertex) - point) * \
                 np.linspace(0, 1, num=CURVE_PLOT_RESOLUTION).reshape((-1, 1))
        points = points[:, [curve_cell.independent_axis, curve_cell.dependent_axis]]  # correct order x, y.
        square = [curve_cell.coords.coords, coords2 if coords2 is not None else curve_cell.coords.coords + 1]
        points_inside_cell = list(map(partial(is_in_square, square=square), points))
        yield points[np.ravel(points_inside_cell), :]


def plot_curve_core(ax, curve_cells):
    for curve_cell in curve_cells:
        # for points in (get_curve_vertex(curve_cell) if
        # isinstance(curve_cell.curve, VertexLinearExtended) else get_curve(curve_cell)):
        for points in get_curve(curve_cell):
            ax.plot(*transform_points2plot(points).T, '-', c=COLOR_CURVE, alpha=1, linewidth=2.5)
