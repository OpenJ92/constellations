from dataclasses import dataclass
from typeclass.data.sequence import Sequence
from typeclass.data.morphism import Morphism

from constellations.geometry.core import SegmentStrip


def _point_equal(a, b, eps: float = 1e-9) -> bool:
    return bool(((a - b) ** 2).sum() ** 0.5 <= eps)


def _dedupe_adjacent(points, eps: float = 1e-9):
    out = []
    for point in points:
        if not out or not _point_equal(out[-1], point, eps=eps):
            out.append(point)
    return out


def _finalize_strip(points, eps: float = 1e-9):
    cleaned = _dedupe_adjacent(points, eps=eps)
    if len(cleaned) < 2:
        return None
    return SegmentStrip(cleaned)


def _append_if_valid(out, points, eps: float):
    built = _finalize_strip(points, eps=eps)
    if built is not None:
        out.append(built)


@dataclass(frozen=True)
class BoundaryPredicate:
    """
    classify(point) must return:
        1  for above / inside
        0  for on boundary
       -1  for below / outside

    intersect(a, b) must return the boundary point when the segment a->b
    crosses or touches the boundary in a meaningful way.
    """
    classify: callable
    intersect: callable
    eps: float = 1e-9


class PartitionBy(Morphism):
    def __init__(self, predicate):
        self.predicate = predicate

    def _run(self, strip: SegmentStrip):
        points = list(strip._values)
        if len(points) < 2:
            return Sequence([]), Sequence([])

        above_out = []
        below_out = []
        above_current = []
        below_current = []

        prev = points[0]
        prev_class = self.predicate.classify(prev)

        if prev_class >= 0:
            above_current.append(prev)
        if prev_class <= 0:
            below_current.append(prev)

        for curr in points[1:]:
            curr_class = self.predicate.classify(curr)
            transition = (prev_class, curr_class)

            if transition in ((1, 1), (1, 0), (0, 0)):
                above_current.append(curr)

            if transition in ((-1, -1), (-1, 0), (0, 0)):
                below_current.append(curr)

            if transition == (0, 1):
                above_current.append(curr)
                _append_if_valid(below_out, below_current, self.predicate.eps)
                below_current = []

            elif transition == (0, -1):
                below_current.append(curr)
                _append_if_valid(above_out, above_current, self.predicate.eps)
                above_current = []

            elif transition == (1, -1):
                boundary = self.predicate.intersect(prev, curr)
                above_current.append(boundary)
                _append_if_valid(above_out, above_current, self.predicate.eps)
                above_current = []
                below_current = [boundary, curr]

            elif transition == (-1, 1):
                boundary = self.predicate.intersect(prev, curr)
                below_current.append(boundary)
                _append_if_valid(below_out, below_current, self.predicate.eps)
                below_current = []
                above_current = [boundary, curr]

            prev = curr
            prev_class = curr_class

        _append_if_valid(above_out, above_current, self.predicate.eps)
        _append_if_valid(below_out, below_current, self.predicate.eps)

        return Sequence(above_out), Sequence(below_out)
