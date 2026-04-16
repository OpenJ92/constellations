from dataclasses import dataclass

from numpy import asarray, min as npmin
from typeclass.data.morphism import Morphism

from constellations.geometry.rectangle import Rectangle
from constellations.morphisms.boundingbox import Box


class Fit(Morphism):
    def __init__(self, target, bbox, margin = 0.0):
        self.target = target
        self.bbox = bbox
        self.margin = margin

    def _run(self, x):
        x = asarray(x, dtype=float)

        target = self.target.inset(self.margin)

        if (target.extent <= 0).any():
            raise ValueError("Target rectangle has non-positive extent after inset")

        if (self.bbox.extent <= 0).any():
            raise ValueError("Bounding box has non-positive extent")

        scale = float(npmin(target.extent / self.bbox.extent))
        source_center = self.bbox.center
        target_center = target.center

        return scale * (x - source_center) + target_center
