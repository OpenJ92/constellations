from numpy import array, linspace

from typeclass.data.morphism import Morphism
from typeclass.data.reader import Reader
from typeclass.typeclasses.symbols import arrow, rcompose, pure, combine, ap, fmap
from typeclass.runtime.core import curry
from typeclass.interpret.run import evaluate

from constellations.geometry.core import SegmentStrip
from constellations.morphisms.translate import Translate 

left  =  Morphism |arrow| (lambda point: point*array([0, 1]))

up    =  Morphism |arrow| Translate(array([1, 0]))

right = (Morphism |arrow| (lambda point: point*array([0, -1])))           \
      |rcompose|                                                          \
        (Morphism |arrow| Translate(array([1, 1])))


down  = (Morphism |arrow| (lambda point: point*array([-1, 0])))           \
      |rcompose|                                                          \
        (Morphism |arrow| Translate(array([1, 0])))


segment = Reader(lambda s: SegmentStrip(linspace(0, 1, s, endpoint=False)))\
close = lambda strip: strip |combine| SegmentStrip((strip._values[0],))
square = Reader |pure| curry(lambda left, up, right, down:                \
               evaluate(left |combine| up |combine| right |combine| down))\
         |ap| (segment |fmap| (lambda s: s |fmap| left))                  \
         |ap| (segment |fmap| (lambda s: s |fmap| up))                    \
         |ap| (segment |fmap| (lambda s: s |fmap| right))                 \
         |ap| (segment |fmap| (lambda s: s |fmap| down))                  \
         |fmap| close
