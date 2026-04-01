from typeclass.data.morphism import Morphism

from constellations.morphisms.sphere import Sphere

class Disk(Morphism):
    def __init__(self):
        pass

    def _run(self, ts):
        t, *ts = ts
        return t * Sphere()(ts)
