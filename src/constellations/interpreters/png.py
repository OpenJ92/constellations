from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from typeclass.data.sequence import Sequence

from constellations.geometry.core import SegmentStrip
from constellations.interpreters.interpreter import Interpreter


@dataclass
class PNG(Interpreter):
    surface: object
    pixels_per_unit: float = 1.0

    background: tuple[int, int, int, int] = (255, 255, 255, 255)
    stroke: tuple[int, int, int, int] = (0, 0, 0, 255)

    stroke_width: float = 1.0
    antialias: int = 2

    @property
    def width_px(self) -> int:
        return max(1, round(self.surface.width * self.pixels_per_unit))

    @property
    def height_px(self) -> int:
        return max(1, round(self.surface.height * self.pixels_per_unit))

    @property
    def scaled_width_px(self) -> int:
        return self.width_px * self.antialias

    @property
    def scaled_height_px(self) -> int:
        return self.height_px * self.antialias

    @property
    def scaled_stroke_width(self) -> int:
        return max(1, round(self.stroke_width * self.pixels_per_unit * self.antialias))

    @property
    def total_scale(self) -> float:
        return self.pixels_per_unit * self.antialias

    def raster_points(self, points):
        return [
            (float(x) * self.total_scale, float(y) * self.total_scale)
            for x, y in points
        ]

    def empty_image(self):
        return Image.new(
            "RGBA",
            (self.scaled_width_px, self.scaled_height_px),
            self.background,
        )

    def run(self, draw, data) -> None:
        match data:

            case SegmentStrip(_values=points):
                if len(points) < 2:
                    return

                draw.line(
                    self.raster_points(points),
                    fill=self.stroke,
                    width=self.scaled_stroke_width,
                    joint="curve",
                )

            case Sequence(_values=values):
                for value in values:
                    self.run(draw, value)

            case _:
                raise NotImplementedError(
                    f"{self.__class__.__name__} has no case for {type(data)}"
                )

    def render(self, data):
        image = self.empty_image()
        draw = ImageDraw.Draw(image)

        self.run(draw, data)

        if self.antialias != 1:
            image = image.resize(
                (self.width_px, self.height_px),
                Image.Resampling.LANCZOS,
            )

        return image

    def write_to_file(self, path: str, data) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        image = self.render(data)
        image.save(path, format="PNG")
