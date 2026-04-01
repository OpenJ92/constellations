from numpy import array, pi
from numpy.random import random

from typeclass.data.stream import Stream, take
from typeclass.data.streamtree import StreamTree, paths
from typeclass.data.morphism import Morphism
from typeclass.data.reader import Reader
from typeclass.interpret.run import evaluate, interpret
from typeclass.typeclasses.symbols import fmap, pure, ap, compose, inverse, arrow, rcompose, identity
from typeclass.runtime.core import curry

from constellations.morphisms.switches import LinearWindow
from constellations.morphisms.translate import Translate
from constellations.morphisms.matrix import Matrix
from constellations.morphisms.rotations import Rotation3D
from constellations.morphisms.sphere import Sphere
from constellations.morphisms.disk import Disk
from constellations.parsers.tree_topology import parser
from constellations.lsystems.tree_topology import lsystem

from .utils import ascending_segment_length_tree, make_offset_from_path, extract

from lsystems.generate import Generate

result = Generate(lsystem, depth=7).run()
tree = parser.run(result)[0][0]

WIDTH, HEIGHT = .2, 20

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
        |fmap| (lambda _: Sphere()(random((3,))))

rotates = StreamTree                                                      \
        |pure| curry(Rotation3D) |ap| axes                                \
        |fmap| (lambda rotate: Morphism |arrow| rotate)

scale   = StreamTree                                                      \
        |pure| None                                                       \
        |fmap| (lambda _: 2*pi*((1/4)*random() - (1/4)))                  \
        |fmap| (lambda angle: Morphism |arrow| (lambda t: angle*t))

rotations = StreamTree                                                    \
        |pure| curry(lambda switch, scale, rotate:                        \
                     switch |rcompose| scale |rcompose| rotate)           \
          |ap| switch                                                     \
          |ap| scale                                                      \
          |ap| rotates
        |fmap| evaluate

root = array((0.0, 0.0))
offset_from_path = make_offset_from_path(1.0, 0.8, 3)
vectors = StreamTree                                                      \
        |pure| curry(lambda root_, offset: root_ + offset)                \
          |ap| (StreamTree |pure| root)                                   \
          |ap| (paths()    |fmap| offset_from_path)                       \
        |fmap| (lambda point: WIDTH*point)

lighthouses = StreamTree                                                  \
        |pure| curry(lambda xy, z: array([*xy, z]))                       \
          |ap| vectors                                                    \
          |ap| (heights |fmap| (lambda h: HEIGHT*h))


translates = lighthouses |fmap| Translate
inverse_translates = lighthouses                                          \
        |fmap| ((Morphism |arrow| inverse) |compose| Translate)

functions = StreamTree                                                    \
        |pure| curry(lambda itra, rot, tra:                               \
              Reader(lambda t: itra |rcompose| rot(t) |rcompose| tra))    \
          |ap| inverse_translates                                         \
          |ap| rotations                                                  \
          |ap| translates 

def composed_functions(streamtree, composed):
    reader = streamtree.value
    children = streamtree.children.force()

    nreader = Reader                                                      \
        |pure| curry(lambda r, q: r |compose| q)                          \
          |ap| composed                                                   \
          |ap| reader                                                     \
        |fmap| evaluate

    nchildren = children                                                  \
        |fmap| (lambda st: composed_functions(st, nreader))

    return StreamTree(evaluate(nreader), interpret(nchildren))

args = evaluate(functions), Reader(lambda _: identity(Morphism))
composed_functions_samples = StreamTree                                   \
        |pure| (lambda vec, func: (vec, func))                            \
          |ap| vectors                                                    \
          |ap| composed_functions(*args)

positions = Stream                                                        \
        |pure| None                                                       \
        |fmap| (lambda _: random((2,)) * array([1, 2*pi]))                \
        |fmap| Disk()                                                     \
        |fmap| (lambda xy: WIDTH*xy)                                      

orientation = Stream                                                      \
        |pure| Matrix(array([[0,0,HEIGHT]]))

locations = positions                                                     \
        |fmap| (lambda xy: array([*xy, 0]))                               \
        |fmap| Translate

lines = Stream                                                            \
        |pure| curry(lambda ori, loc: ori |rcompose| loc)                 \
          |ap| orientation                                                \
          |ap| locations                                                  \
        |fmap| evaluate

line_samples = Stream                                                     \
        |pure| curry(lambda pos, line: (pos, line))                       \
          |ap| positions                                                  \
          |ap| lines
