from dataclasses import dataclass

from numpy import array

from constellations.geometry.rectangle import Rectangle

@dataclass(frozen=True)
class Paper:
    width: float
    height: float
    unit: str = "mm"
    name: str = ""

    @property
    def rectangle(self) -> Rectangle:
        return Rectangle(
            array([0.0, 0.0], dtype=float),
            array([self.width, self.height], dtype=float),
        )

A0x2 = Paper(1189.0, 1682.0, name="A0x2")
A0   = Paper(841.0, 1189.0, name="A0")
A1   = Paper(594.0, 841.0 , name="A1")
A2   = Paper(420.0, 594.0 , name="A2")
A3   = Paper(297.0, 420.0 , name="A3")
A4   = Paper(210.0, 297.0 , name="A4")
A5   = Paper(148.0, 210.0 , name="A5")
