from numpy import array, pi
from numpy.random import random, default_rng
from numpy.linalg import norm

from typeclass.data.streamtree import StreamTree, paths, coordinates, depths
from typeclass.data.morphism import Morphism
from typeclass.data.reader import Reader
from typeclass.interpret.run import evaluate
from typeclass.typeclasses.symbols import fmap, pure, ap, compose, inverse, arrow, rcompose
from typeclass.runtime.core import curry

from constellations.morphisms.switches import LinearWindow, SmoothWindow
from constellations.morphisms.translate import Translate
from constellations.morphisms.rotations import Rotation3D
from constellations.parsers.tree_topology import parser
from constellations.lsystems.tree_topology import lsystem

from .utils import ascending_segment_length_tree, make_offset_from_path, extract

from lsystems.generate import Generate


result = Generate(lsystem, depth=7).run()
tree = parser.run(result)[0][0]

heights = ascending_segment_length_tree()
widths  = StreamTree                                                      \
        |pure| None                                                       \
        |fmap| (lambda _: random())                                       \
        |fmap| (lambda x: 1/50*x)

switch  = StreamTree                                                      \
        |pure| curry(LinearWindow)                                        \
          |ap| heights                                                    \
          |ap| widths  

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

rotations = StreamTree                                                    \
        |pure| curry(lambda switch, scale, rotate, t:                     \
                     switch |rcompose| scale |rcompose| rotate(t))        \
          |ap| switch                                                     \
          |ap| scale                                                      \
          |ap| rotates

root = array((0.0, 0.0))
offset_from_path = make_offset_from_path(1.0, 0.8, 3)
translates = StreamTree                                                   \
        |pure| curry(lambda root_, offset: root_ + offset)                \
          |ap| (StreamTree |pure| root)                                   \
          |ap| (paths() |fmap| offset_from_path)

lighthouses = StreamTree\
        |pure| curry(lambda xy, z: array([*xy, z]))\
          |ap| translates\
          |ap| heights


translates = lighthouses |fmap| Translate
inverse_translates = lighthouses \
                   |fmap| ((Morphism |arrow| inverse) |compose| Translate)

functions = StreamTree\
        |pure| curry(lambda itra, rot, tra:\
                     Reader(lambda t: itra |rcompose| rot(t) |rcompose| tra))\
          |ap| inverse_translates\
          |ap| rotations\
          |ap| translates\
          ## |ap| (StreamTree |pure| 0)
