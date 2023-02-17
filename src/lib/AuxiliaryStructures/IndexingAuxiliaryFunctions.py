from numbers import Integral
from typing import List, Tuple, Union

import numpy as np


class CellCoords:
    def __init__(self, coords: Union[Tuple[int], np.ndarray]):
        self.coords = np.array(coords)

    @property
    def tuple(self):
        return tuple(self.coords)

    @property
    def array(self):
        return self.coords

    def __str__(self):
        return str(self.tuple)

    def __getitem__(self, item):
        assert isinstance(item, Integral), "Only integer indexing valid."
        return self.coords[item]

    def __setitem__(self, key, value):
        self.coords[key] = value

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return CellCoords(self.coords + other)
        elif isinstance(other, CellCoords):
            return CellCoords(self.coords + other.coords)
        elif isinstance(other, (np.ndarray, List, Tuple)):
            if len(np.shape(other)) == 2:
                return [self + o for o in other]
            else:
                return CellCoords(self.coords + other)
        else:
            raise Exception("Not implemented addition for type {}".format(type(other)))

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return CellCoords(self.coords - other)
        elif isinstance(other, CellCoords):
            return CellCoords(self.coords - other.coords)
        elif isinstance(other, (np.ndarray, List, Tuple)):
            if len(np.shape(other)) == 2:
                return [self - o for o in other]
            else:
                return CellCoords(self.coords - other)
        else:
            raise Exception("Not implemented addition for type {}".format(type(other)))
