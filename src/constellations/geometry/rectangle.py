from dataclasses import dataclass

from numpy import array


@dataclass(frozen=True)
class Rectangle:
    min: array
    max: array

    @property
    def extent(self):
        return self.max - self.min

    @property
    def center(self):
        return (self.min + self.max) / 2.0

    @property
    def width(self) -> float:
        return float(self.extent[0])

    @property
    def height(self) -> float:
        return float(self.extent[1])

    def inset(self, margin: float) -> "Rectangle":
        offset = array([margin, margin], dtype=float)
        return Rectangle(self.min + offset, self.max - offset)
