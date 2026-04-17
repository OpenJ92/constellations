from perlin_noise import PerlinNoise
from numpy import array, square
from numpy.random import randint, rand

from typeclass.data.morphism import Morphism
from constellations.morphisms.sphere import Sphere


## Rn -> R1
class Perlin_Noise(Morphism):
    def __init__(self, octave, seed, scale=1):
        self.octave = octave
        self.seed = seed
        self.scale = scale
        self.perline_noise = PerlinNoise(self.octave, self.seed)

    def _run(self, ts: array):
        return self.scale * self.perline_noise(ts)


## Rn -> R1
class Perlin_Stack(Morphism):
    def __init__(self, proportions, octaves, seeds):
        self.proportions = proportions
        self.octaves = octaves
        self.seeds = seeds

    def _run(self, ts: array):
        perlins  = []
        traverse = zip(self.octaves, self.seeds, self.proportions)
        for octave, seed, proportion in traverse:
            perlins.append(Perlin_Noise(octave, int(seed), proportion))

        retval = 0
        for perlin in perlins:
            retval += perlin(ts)
        return retval


Perlin = Perlin_Noise | Perlin_Stack


## Rn -> Rm transform
class Perlin_Vector(Morphism):
    def __init__(self, perlins):
        self.perlins = perlins

    def _run(self, ts: array):
        return array(list(map(lambda perlin: perlin(ts), self.perlins)))
