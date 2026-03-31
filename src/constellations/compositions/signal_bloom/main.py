from hashlib import sha256
from numpy import array, pi, exp, cos, sin
from numpy.random import random, default_rng
from numpy.linalg import norm

from typeclass.data.streamtree import StreamTree, paths, coordinates, depths
from typeclass.data.morphism import Morphism
from typeclass.interpret.run import evaluate
from typeclass.typeclasses.symbols import fmap, pure, ap, compose, inverse, arrow
from typeclass.runtime.core import curry


from constellations.morphisms.switches import LinearWindow, SmoothWindow
from constellations.morphisms.translate import Translate
from constellations.morphisms.rotations import Rotation3D
from constellations.parsers.tree_topology import parser
from constellations.lsystems.tree_topology import lsystem

from .utils import ascending_segment_length_tree, make_offset_from_path


from lsystems.generate import Generate


result = Generate(lsystem, depth=7).run()
tree = parser.run(result)[0][0]

_centers = ascending_segment_length_tree()
_widths  = StreamTree                                                     \
        |pure| None                                                       \
        |fmap| (lambda _: random())                                       \
        |fmap| (lambda x: 1/50*x)

switch  = StreamTree                                                      \
        |pure| curry(LinearWindow)                                        \
          |ap| _centers                                                   \
          |ap| _widths

axes    = StreamTree                                                      \
        |pure| None                                                       \
        |fmap| (lambda _: random((3,)) - array([.5, .5, .5]))

rotates = StreamTree                                                      \
        |pure| curry(Rotation3D) |ap| axes                                \
        |fmap| (lambda rotate: Morphism |arrow| rotate)

scale   = StreamTree                                                      \
        |pure| None                                                       \
        |fmap| (lambda _: 2*pi*(2*random() - 1))                          \
        |fmap| (lambda angle: Morphism |arrow| (lambda t: angle*t))

expr    = StreamTree                                                      \
        |pure| curry(lambda switch, scale, rotate:                        \
                     evaluate(switch |rcompose| scale |rcompose| rotate)) \
          |ap| switch                                                     \
          |ap| scale                                                      \
          |ap| rotates

root = array((0.0, 0.0))
offset_from_path = make_offset_from_path(1.0, 0.8, 3)

centers = StreamTree                                                      \
        |pure| curry(lambda root_, offset: root_ + offset)                \
        |ap| (StreamTree |pure| root)                                     \
        |ap| (paths() |fmap| offset_from_path)


                        
lighthouses = centers                                                     \
            |fmap| (lambda pair: array([*pair, 0]))

translates = lighthouses |fmap| Translate
inverses   = lighthouses |fmap| ((Morphism |arrow| inverse) |compose| Translate)
