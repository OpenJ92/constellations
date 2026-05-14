from dataclasses import dataclass
from pathlib import Path

from typeclass.data.sequence import Sequence

from constellations.geometry.core import SegmentStrip
from constellations.interpreters.interpreter import Interpreter


@dataclass
class SVG(Interpreter):
    surface: object
    stroke: str = "black"
    fill: str = "none"
    stroke_width: float = 1.0

    @property
    def width(self) -> str:
        return f"{self.surface.width}{self.surface.unit}"

    @property
    def height(self) -> str:
        return f"{self.surface.height}{self.surface.unit}"

    @property
    def viewBox(self) -> str:
        return f"0 0 {self.surface.width} {self.surface.height}"

    @property
    def default_meta(self) -> str:
        return (
            f'fill="{self.fill}" '
            f'stroke="{self.stroke}" '
            f'stroke-width="{self.stroke_width}"'
        )

    def svg_points(self, points) -> str:
        return " ".join(f"{x:.4f},{y:.4f}" for x, y in points)

    def run(self, data) -> str:
        match data:

            case SegmentStrip(_values=points):
                if len(points) < 2:
                    return ""

                pts = self.svg_points(points)
                return f'<polyline points="{pts}" {self.default_meta}/>\n'

            case Sequence(_values=values):
                return "".join(self.run(value) for value in values)

            case _:
                raise NotImplementedError(
                    f"{self.__class__.__name__} has no case for {type(data)}"
                )

    def wrap(self, work: str) -> str:
        return (
            f'<svg width="{self.width}" height="{self.height}" '
            f'viewBox="{self.viewBox}" '
            f'xmlns="http://www.w3.org/2000/svg">\n'
            f'{work}'
            f'</svg>\n'
        )

    def render(self, data) -> str:
        return self.wrap(self.run(data))

    def write_to_file(self, path: str, data) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            file.write(self.render(data))
