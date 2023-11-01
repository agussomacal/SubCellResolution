from typing import Tuple, Dict, Generator, List, Callable

import numpy as np

from lib.AuxiliaryStructures.Constants import NEIGHBOURHOOD_8_MANHATTAN, NEIGHBOURHOOD_16_MANHATTAN
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords, ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import CellBase, CURVE_CELL_TYPE, VERTEX_CELL_TYPE
from lib.CellCreators.CurveCellCreators.CurveCellCreatorBase import CurveCellCreatorBase, CellCurveBase
from lib.Curves.CurvePolynomial import CurvePolynomial
from lib.Curves.CurveVertex import CurveVertexLinearAngle
from lib.Curves.Curves import Curve
from lib.StencilCreators import Stencil


def vertex_in_cell(coords, independent_axis, vertex):
    return (coords[independent_axis] <= vertex[0] <= coords[independent_axis] + 1) and (
            coords[1 - independent_axis] <= vertex[1] <= coords[1 - independent_axis] + 1)


class VertexLinearExtended(CurveVertexLinearAngle):
    def __init__(self, point1, vertex, point2, angle1, angle2, x0: float, y0: float, value_up, value_down):
        super().__init__(angle1, angle2, x0, y0, value_up, value_down)
        self.point1 = point1
        self.vertex = vertex
        self.point2 = point2


# --------------------------------------------------- #
# ------------- Curve cell creator base ------------- #
def get_neighbouring_singular_coords_under_condition(coords: CellCoords, indexer: ArrayIndexerNd,
                                                     condition: Callable, cells, max_num_nodes=5):
    """
    Looks around the vertex searching for curve cells that go opposite directions from the vertex.
    :param coords:
    :param indexer:
    :param cell_mask:
    :param condition:
    :param max_num_nodes:
    :return:
    """
    # look in the 2 distance neighborhood.
    visited_nodes = [
        indexer[new_coords.coords] for new_coords in coords + NEIGHBOURHOOD_16_MANHATTAN
        if indexer[new_coords.tuple] in cells and cells[indexer[new_coords.tuple]].CELL_TYPE == CURVE_CELL_TYPE
    ]

    singular_cells = []
    if len(visited_nodes) >= 2:
        nodes2visit = [[visited_nodes.pop(0)]]
        visited_nodes = sorted(visited_nodes, key=lambda c: np.sum(np.abs(np.array(c) - np.array(nodes2visit[0][0]))))
        nodes2visit.append([visited_nodes.pop(-1)])
        # add as visited the nodes neighbouring the cell
        visited_nodes = visited_nodes + [
            indexer[new_coords.coords] for new_coords in coords + NEIGHBOURHOOD_8_MANHATTAN
            if indexer[new_coords.tuple] in cells and
                                                         cells[indexer[new_coords.tuple]].CELL_TYPE == CURVE_CELL_TYPE
        ] + [coords.tuple]
        stop = [False, False]
        for ix in range(max_num_nodes):
            for direction in [0, 1]:
                if not stop[direction]:
                    visiting_coords = nodes2visit[direction].pop(0)
                    visited_nodes.append(visiting_coords)
                    if condition(visiting_coords):
                        singular_cells.append(CellCoords(visiting_coords))
                        stop[direction] = True
                    else:
                        new_node_candidates = {
                            indexer[new_coords.tuple] for new_coords in
                            CellCoords(visiting_coords) + NEIGHBOURHOOD_8_MANHATTAN
                            if indexer[new_coords.tuple] in cells and
                            cells[indexer[new_coords.tuple]].CELL_TYPE == CURVE_CELL_TYPE
                        }
                        # only adds new coordinates that havent been seen before neither in the others surfer list.
                        nodes2visit[direction] += list(
                            new_node_candidates.difference(
                                visited_nodes + nodes2visit[0] + nodes2visit[1]))
                        if len(nodes2visit[direction]) == 0:
                            stop[direction] = True
            if all(stop):
                break
    return singular_cells


def get_neighbouring_singular_cells(coords: CellCoords, cell_mask: np.ndarray, indexer: ArrayIndexerNd,
                                    cells: Dict[Tuple[int], CellBase], stencils: Dict[Tuple, List[Tuple[int, ...]]]):
    # the condition is to search cells at least at 2 of distance (not the immediate ones.
    singular_neighbours_stencil = [
        cell_coords for cell_coords in
        get_neighbouring_singular_coords_under_condition(
            coords, indexer, max_num_nodes=10, cells=cells,
            condition=lambda coordinates: coords.tuple not in stencils[coordinates] and
                                          coordinates in cells and
                                          cells[coordinates].CELL_TYPE == CURVE_CELL_TYPE)
        if cell_coords.tuple in cells and cells[cell_coords.tuple].CELL_TYPE == CURVE_CELL_TYPE]
    # if len(singular_neighbours_stencil) >= 5:
    if len(singular_neighbours_stencil) == 2:
        return tuple([cells[c.tuple] for c in singular_neighbours_stencil[-2:]])
    else:
        return None


class VertexCellCreatorUsingNeighbours(CurveCellCreatorBase):
    def create_vertex(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple,
                      singular_neighbours: Tuple[CellCurveBase, CellCurveBase], **kwargs) -> Generator[
        Curve, None, None]:
        raise Exception("Not implemented.")

    def create_cells(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                     coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                     stencil: Stencil, stencils: Dict[Tuple[int, ...], np.ndarray]) -> Generator[CellBase, None, None]:
        regular_opposite_cells = self.regular_opposite_cell_searcher(
            coords=coords, independent_axis=independent_axis, average_values=average_values,
            smoothness_index=smoothness_index, indexer=indexer, cells=cells, stencil=stencil,
            stencils=stencils
        )

        # TODO: smoothness_index>0 look for other way.
        singular_neighbours = get_neighbouring_singular_cells(coords=coords,
                                                              cell_mask=smoothness_index > 0,
                                                              indexer=indexer, cells=cells,
                                                              stencils=stencils)
        if len(regular_opposite_cells) == 2 and singular_neighbours is not None:
            for curve in self.create_vertex(average_values=average_values, indexer=indexer, cells=cells, coords=coords,
                                            smoothness_index=smoothness_index, independent_axis=independent_axis,
                                            stencil=stencil, regular_opposite_cells=regular_opposite_cells,
                                            singular_neighbours=singular_neighbours):

                # TODO: specify a new class?!!
                if not isinstance(curve, CurvePolynomial):
                    new_coords = [coords.tuple] + [
                        indexer[new_coords.coords] for new_coords in coords + NEIGHBOURHOOD_8_MANHATTAN
                        if indexer[new_coords.tuple] in cells and cells[indexer[
                            new_coords.tuple]].CELL_TYPE == CURVE_CELL_TYPE
                    ]
                    for c in new_coords:
                        cell = CellCurveBase(
                            coords=CellCoords(c),
                            curve=curve,
                            regular_opposite_cells=regular_opposite_cells,
                            dependent_axis=1 - independent_axis)
                        cell.CELL_TYPE = VERTEX_CELL_TYPE
                        yield cell, CellCoords(c)
                else:
                    yield CellCurveBase(
                        coords=coords,
                        curve=curve,
                        regular_opposite_cells=regular_opposite_cells,
                        dependent_axis=1 - independent_axis)


def eval_neighbour_in_border(coords: CellCoords, singular_neighbour: CellCurveBase, independent_axis):
    versor = singular_neighbour.coords.array - coords.array
    crossing_axis = np.ravel(np.where(versor))[0]
    crossing = coords[crossing_axis] + 0.5 + np.sign(versor[crossing_axis]) / 2
    if singular_neighbour.independent_axis == crossing_axis:
        # TODO: evaluation is not a value, needs an array, what to do with nans, filter before?
        evals = np.ravel(singular_neighbour.curve(np.array([crossing])))
        point = (crossing, evals[~np.isnan(evals)][0])
    else:
        inverse = np.ravel(np.array(singular_neighbour.curve.function_inverse(crossing)))
        if len(inverse) == 0:  # when there is no intersection
            return None, None
        point = (inverse[~np.isnan(inverse)][0], crossing)
    der_evals = np.ravel(singular_neighbour.curve.derivative(point[0]))
    der_evals = der_evals[~np.isnan(der_evals)]
    if len(der_evals) >= 1:
        versor = (1, der_evals[0])
    else:
        versor = (1, 1)  # TODO: this is nothing good, only to avoid crashing here

    same_independent_axis = int(independent_axis == singular_neighbour.independent_axis)
    return np.array(point)[[1 - same_independent_axis, same_independent_axis]], \
           np.array(versor)[[1 - same_independent_axis, same_independent_axis]]


def vector2angle(vector):
    vector = vector / np.linalg.norm(vector)
    angle = np.arccos(vector[0])
    return np.pi * 2 * (vector[1] < 0) + ((vector[1] >= 0) * 2 - 1) * angle


class VertexCellCreatorUsingNeighboursLines(VertexCellCreatorUsingNeighbours):
    def create_vertex(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple,
                      singular_neighbours: Tuple[CellCurveBase, CellCurveBase], **kwargs) -> Generator[
        Curve, None, None]:
        point1, versor1 = eval_neighbour_in_border(coords, singular_neighbours[0], independent_axis)
        point2, versor2 = eval_neighbour_in_border(coords, singular_neighbours[1], independent_axis)

        if point1 is not None and point2 is not None:  # happens if no intersection of curve with cell
            a = np.transpose([versor1, -versor2])
            if np.linalg.det(a) == 0:
                vertex = (point1 + point2) / 2
            else:
                l = np.linalg.solve(a=np.transpose([versor1, -versor2]), b=point2 - point1)
                vertex = point1 + l[0] * versor1
            angle1 = vector2angle(point1 - vertex)
            angle2 = vector2angle(point2 - vertex)
            # TODO: the NaN appears when point1==point2, avoid here or where?
            if not np.isnan(angle2 + angle1):
                value_up = regular_opposite_cells[1].evaluate(coords.coords)
                value_down = regular_opposite_cells[0].evaluate(coords.coords)
                yield VertexLinearExtended(
                    point1=point1,
                    vertex=vertex,
                    point2=point2,
                    angle1=angle1,
                    angle2=angle2 - angle1,
                    x0=vertex[0], y0=vertex[1],
                    value_up=value_up,
                    value_down=value_down
                )


if __name__ == "__main__":
    fx = lambda theta, alpha, beta, L: np.tan(beta) * L / (np.tan(alpha - theta) - np.tan(beta - theta))
    fy = lambda theta, alpha, beta, L: fx(theta, alpha, beta, L) * np.tan(alpha - theta)

    theta = np.linspace(-np.pi, np.pi)
    alpha = np.pi * 30 / 180
    beta = np.pi * 60 / 180
    L = 1

    import matplotlib.pylab as plt

    plt.close("all")
    plt.plot(fx(theta, alpha, beta, L), fy(theta, alpha, beta, L))
    plt.show()
