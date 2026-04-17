from numpy import diag

from typeclass.typeclasses.symbols import fmap, pure, ap
from typeclass.interpret.run import evaluate
from typeclass.runtime.core import curry
from typeclass.data.reader import Reader

from constellations.morphisms.translate import Translate
from constellations.morphisms.matrix import Matrix

from constellations.realizations.primitives.square import square

rectangle = Reader(lambda rect:
    square
    |fmap| (lambda strip: strip |fmap| Matrix(diag(rect.extent)))
    |fmap| (lambda strip: strip |fmap| Translate(rect.min))
)
