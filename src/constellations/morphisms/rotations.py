from typeclass.data.automorphism import Automorphism

from numpy import array, sin, cos, dot
from numpy.linalg import norm, cross

class Rotation3D(Automorphism):
    def __init__(self, axis, angle):
        axis = array(axis, dtype=float)
        n = norm(axis)
        if n == 0:
            raise ValueError("Rotation axis must be non-zero.")
        self.axis = axis / n
        self.angle = angle

    def _run(self, x):
        k = self.axis
        theta = self.angle

        return (
            x * cos(theta)
            + cross(k, x) * sin(theta)
            + k * dot(k, x) * (1 - cos(theta))
        )

    def _inv(self):
        return Rotation3D(self.axis, -self.angle)
