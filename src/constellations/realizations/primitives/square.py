from numpy import array

from typeclass.data.morphism import Morphism
from typeclass.data.reader import Reader
from typeclass.typeclasses.symbols import arrow, rcompose, pure, combine, ap, fmap
from typeclass.runtime.core import curry
from typeclass.interpret.run import evaluate

from constellations.geometry.core import SegmentStrip
from constellations.morphisms.translate import Translate 

from constellations.realizations.primitives.segment import segment

left = Morphism |arrow| (lambda t: array([0.0, t]))

up = Morphism |arrow| (lambda t: array([t, 1.0]))

right = Morphism |arrow| (lambda t: array([1.0, 1.0 - t]))

down = Morphism |arrow| (lambda t: array([1.0 - t, 0.0]))


close = lambda strip: strip |combine| SegmentStrip((strip._values[0],))
square = Reader |pure| curry(lambda left, up, right, down:                \
               evaluate(left |combine| up |combine| right |combine| down))\
         |ap| (segment |fmap| (lambda s: s |fmap| left))                  \
         |ap| (segment |fmap| (lambda s: s |fmap| up))                    \
         |ap| (segment |fmap| (lambda s: s |fmap| right))                 \
         |ap| (segment |fmap| (lambda s: s |fmap| down))                  \
         |fmap| close

Square = evaluate(square)
