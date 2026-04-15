from numpy import diag

from typeclass.typeclasses.symbols import fmap, pure, ap
from typeclass.interpret.run import evaluate
from typeclass.runtime.core import curry
from typeclass.data.reader import Reader

from constellations.morphisms.translate import Translate
from constellations.morphisms.matrix import Matrix

rectangle = Reader |pure| curry(lambda rectangle, square:                 \
    square                                                                \
        |fmap| Matrix(diag(rectangle.extent))                             \
        |fmap| Translate(rectangle.min))

