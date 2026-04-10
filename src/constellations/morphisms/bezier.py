from numpy import asarray, ndarray, arange, sum, indices, moveaxis
from math import comb

from typeclass.data.morphism import Morphism


class Bezier(Morphism):
    def __init__(self, control_points, collapse_axes):
        self.control_points = asarray(control_points, dtype=float)
        self.collapse_axes = tuple(collapse_axes)

        self._binoms = tuple(
            asarray(
                [comb(self.control_points.shape[axis] - 1, i)
                 for i in range(self.control_points.shape[axis])],
                dtype=float,
            )
            for axis in self.collapse_axes
        )

    @staticmethod
    def _basis(n: int, t: float, binoms: ndarray) -> ndarray:
        i = arange(n + 1, dtype=float)
        return binoms * ((1.0 - t) ** (n - i)) * (t ** i)

    @staticmethod
    def _collapse_axis(points: ndarray, axis: int, basis: ndarray) -> ndarray:
        shape = [1] * points.ndim
        shape[axis] = points.shape[axis]
        basis = basis.reshape(shape)
        return sum(points * basis, axis=axis)

    def _run(self, ts) -> ndarray:
        ts = asarray(ts, dtype=float)
        if len(ts) != len(self.collapse_axes):
            raise ValueError("Bezier expected one parameter per collapse axis")

        points = self.control_points
        axes = list(self.collapse_axes)

        for i, t in enumerate(ts):
            axis = axes[i]
            n = points.shape[axis] - 1
            basis = self._basis(n, t, self._binoms[i])
            points = self._collapse_axis(points, axis, basis)
            axes = [a if a < axis else a - 1 for a in axes]

        return points

    @classmethod
    def ID(cls, k: int):
        shape = (2,) * k
        grid = indices(shape, dtype=float)
        control_points = moveaxis(grid, 0, -1)   # (2,)*k + (k,)
        collapse_axes = tuple(range(k - 1, -1, -1))
        return cls(control_points, collapse_axes)
