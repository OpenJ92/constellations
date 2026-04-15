from dataclasses import dataclass

from numpy import asarray, minimum, maximum
from typeclass.data.morphism import Morphism
from typeclass.data.sequence import Sequence

from constellations.geometry.core import SegmentStrip


@dataclass(frozen=True)
class Box:
    min: array
    max: array

    @classmethod
    def point(cls, point) -> Box:
        point = asarray(point, dtype=float)
        return cls(point.copy(), point.copy())

    def combine(self, other: Box) -> Box:
        return Box(
            minimum(self.min, other.min),
            maximum(self.max, other.max),
        )

    @property
    def extent(self):
        return self.max - self.min

    @property
    def center(self):
        return (self.min + self.max) / 2.0

    @property
    def dim(self) -> int:
        return int(self.min.shape[0])


class BoundingBox(Morphism):
    def __init__(self):
        pass

    def _run(self, data) -> Box:
        match data:
            case SegmentStrip(_values=points):
                points = tuple(points)

                if not points:
                    raise ValueError("Cannot compute bounding box of empty SegmentStrip")

                box = Box.point(points[0])
                for point in points[1:]:
                    box = box.combine(Box.point(point))
                return box

            case Sequence(_values=values):
                values = tuple(values)

                if not values:
                    raise ValueError("Cannot compute bounding box of empty Sequence")

                box = self._run(values[0])
                for value in values[1:]:
                    box = box.combine(BoundingBox()(value))
                return box

            case _:
                raise NotImplementedError(
                    f"{self.__class__.__name__} has no case for {type(data)}"
                )
