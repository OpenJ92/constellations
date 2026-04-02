from dataclasses import dataclass
import numpy as np

from constellations.morphisms.partitionby.core import PartitionBy, BoundaryPredicate


@dataclass(frozen=True)
class Plane2D:
    normal: np.ndarray
    anchor: np.ndarray
    eps: float = 1e-9

    def side(self, point) -> float:
        return float(np.dot(self.normal, point - self.anchor))

    def classify(self, point) -> int:
        s = self.side(point)
        if s > self.eps:
            return 1
        if s < -self.eps:
            return -1
        return 0

    def intersect(self, a, b):
        sa = self.side(a)
        sb = self.side(b)
        denom = sa - sb
        if abs(denom) <= self.eps:
            return a.copy()
        t = sa / denom
        return a + t * (b - a)


class PartitionByPlane(PartitionBy):
    def __init__(self, plane: Plane2D):
        super().__init__(
            BoundaryPredicate(
                classify=plane.classify,
                intersect=plane.intersect,
                eps=plane.eps,
            )
        )
        self.plane = plane
