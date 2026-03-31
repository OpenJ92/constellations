from typeclass.data.morphism import Morphism

from numpy import exp

class SmoothWindow(Morphism):
    def __init__(self, center, width):
        if width <= 0:
            raise ValueError("width must be positive.")
        self.start = center - width / 2
        self.end = center + width / 2

    def _run(self, x):
        t = (x - self.start) / (self.end - self.start)
        if t < 0:
            t = 0.0
        elif t > 1:
            t = 1.0
        return t * t * (3 - 2 * t)

class LinearWindow(Morphism):
    def __init__(self, center, width):
        if width <= 0:
            raise ValueError("width must be positive.")
        self.start = center - width / 2
        self.end = center + width / 2

    def _run(self, x):
        t = (x - self.start) / (self.end - self.start)
        if t < 0:
            return 0.0
        if t > 1:
            return 1.0
        return t

class SmoothInterval(Morphism):
    def __init__(self, start, end):
        if start == end:
            raise ValueError("start and end must differ.")
        self.start = start
        self.end = end

    def _run(self, x):
        t = (x - self.start) / (self.end - self.start)
        if t < 0:
            t = 0.0
        elif t > 1:
            t = 1.0
        return t * t * (3 - 2 * t)

class LinearInterval(Morphism):
    def __init__(self, start, end):
        if start == end:
            raise ValueError("start and end must differ.")
        self.start = start
        self.end = end

    def _run(self, x):
        t = (x - self.start) / (self.end - self.start)
        if t < 0:
            return 0.0
        if t > 1:
            return 1.0
        return t

class SigmoidInterval(Morphism):
    def __init__(self, center, sharpness=10.0):
        self.center = center
        self.sharpness = sharpness

    def _run(self, x):
        return 1.0 / (1.0 + exp(-self.sharpness * (x - self.center)))
