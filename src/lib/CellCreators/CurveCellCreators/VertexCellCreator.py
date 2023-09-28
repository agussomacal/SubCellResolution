import os
from typing import Generator, Tuple, Dict

import numpy as np

from PerplexityLab.miscellaneous import timeit
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords, ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import CellBase
from lib.CellCreators.CurveCellCreators.CurveCellCreatorBase import CurveCellCreatorBase, \
    prepare_stencil4one_dimensionalization
from lib.Curves.CurveVertex import CurveVertexPolynomial
from lib.Curves.Curves import Curve
from lib.StencilCreators import Stencil

ROUND2MACHINE_PRECISION = 8

path2main_equations = f"{os.path.dirname(os.path.abspath(__file__))}/VertexMainEquations.py"
if not os.path.exists(path2main_equations):
    import sympy as sp

    # Averages
    S0 = sp.Symbol("S0", negative=False, real=True)
    S1 = sp.Symbol("S1", negative=False, real=True)
    S2 = sp.Symbol("S2", negative=False, real=True)
    S3 = sp.Symbol("S3", negative=False, real=True)

    # Curve parameters
    ml = sp.Symbol(r"m_1")
    mr = sp.Symbol(r"m_2")
    bl = sp.Symbol(r"b_1")
    br = sp.Symbol(r"b_2")
    xv = sp.Symbol(r"x_v", positive=True)  # x of vertex

    # Independent variable
    x = sp.Symbol("x")
    y = sp.Symbol("y")

    # Functions
    fl = ml * x + bl
    fr = mr * x + br
    Fl = sp.integrate(fl, x)
    Fr = sp.integrate(fr, x)

    # Equations for cases:
    Sl = lambda xi, xf: Fl.subs(x, xf) - Fl.subs(x, xi)
    Sr = lambda xi, xf: Fr.subs(x, xf) - Fr.subs(x, xi)

    vertex_eq = [sp.Eq(fl.subs(x, xv), fr.subs(x, xv))]
    eq23 = [sp.Eq(Sl(0, 1), S0)]
    eq_if_vertex_in_1 = [sp.Eq(Sl(1, xv) + Sr(xv, 2), S1), sp.Eq(Sr(2, 3), S2)]
    eq_if_vertex_in_2 = [sp.Eq(Sl(1, 2), S1), sp.Eq(Sl(2, xv) + Sr(xv, 3), S2)]
    eq67 = [sp.Eq(Sr(3, 4), S3)]

    # A->B cases
    equations = {
        "2,3->6,7v1": eq23 + eq67 + eq_if_vertex_in_1 + vertex_eq,
        "2,3->6,7v2": eq23 + eq67 + eq_if_vertex_in_2 + vertex_eq,
    }

    with open(path2main_equations, "w") as f:
        f.writelines([
            "import numpy as np\n",
            # "import numba\n",
            # "@numba.njit()\n",
            "def main_equations(S0, S1, S2, S3):\n",  # , S0I, S1I, S2I, S3I, S4I
            "\tpolynomials=[]\n",
            "\tvertices=[]\n\n"
        ])

        for k, eq in equations.items():
            with timeit(f"Solving equations: {k}"):
                solution = sp.solve(
                    eq,
                    [bl, ml, br, mr, xv]
                )
            print(f"Creating script for equations: {k}, number of solutions {len(solution)}")
            for s in solution:
                s = list(map(lambda r: str(sp.sympify(r)).replace("sqrt", "np.sqrt"), s))
                # f.write(f"\tyield ([{s[0]}, {s[1]}],[{s[2]}, {s[3]}]), {s[4]}", end="\n\n")
                f.writelines([
                    "\ttry:\n",
                    f"\t\tpolynomials.append(([{s[0]}, {s[1]}],[{s[2]}, {s[3]}]))\n",
                    f"\t\tvertices.append({s[4]})\n",
                    "\texcept:\n",
                    "\t\tvalid_len = min(len(polynomials), len(vertices))\n",
                    "\t\tpolynomials = polynomials[:valid_len]\n",
                    "\t\tvertices = vertices[:valid_len]\n\n",
                ])

        f.writelines([
            "\treturn polynomials, vertices\n"
        ])

from lib.CellCreators.CurveCellCreators.VertexMainEquations import main_equations


class LinearVertexCellCurveCellCreator(CurveCellCreatorBase):
    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[Curve, None, None]:
        value_up, value_down = self.updown_value_getter(coords, regular_opposite_cells)
        stencil_values = prepare_stencil4one_dimensionalization(independent_axis, value_up, value_down, stencil,
                                                                smoothness_index, indexer)
        stencil_values = stencil_values.sum(axis=1)

        x_shift = np.min(stencil.coords[:, independent_axis])
        y_shift = np.min(stencil.coords[:, 1 - independent_axis])
        if len(stencil_values) == 4:
            try:
                polynomials, vertices = main_equations(*stencil_values)
            except TypeError:
                print("Can not unpack")
                polynomials = vertices = []
            for ps, x0 in zip(polynomials, np.round(vertices, decimals=ROUND2MACHINE_PRECISION)):
                if ~np.isnan(x0) and ~np.any(np.isnan(polynomials)):
                    ps[0][0] += y_shift
                    ps[1][0] += y_shift
                    yield CurveVertexPolynomial(
                        polynomials=np.round(ps, decimals=ROUND2MACHINE_PRECISION),
                        x0=x0,
                        value_up=value_up,
                        value_down=value_down,
                        x_shift=x_shift
                    )
