from dataclasses import dataclass
import numpy as np

from constellations.morphisms.partitionby.core import PartitionBy, BoundaryPredicate

@dataclass(frozen=True)
class Line2D:
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

class PartitionByLine(PartitionBy):
    def __init__(self, line):
        super().__init__(
            BoundaryPredicate(
                classify=line.classify,
                intersect=line.intersect,
                eps=line.eps,
            )
        )
        self.line = line
