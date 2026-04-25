# ================================
# Imports
# ================================

from numpy import array, pi, linspace, diag
from numpy.random import default_rng, randint
from numpy.linalg import norm
from functools import lru_cache
from hashlib import sha256
import time

from typeclass.data.sequence import Sequence
from typeclass.data.stream import Stream, take, iterate
from typeclass.data.streamtree import StreamTree, paths, depths
from typeclass.data.morphism import Morphism
from typeclass.data.reader import Reader
from typeclass.data.tree import pretty
from typeclass.interpret.run import evaluate, interpret
from typeclass.typeclasses.symbols import (
    fmap, pure, ap, compose, inverse, arrow, rcompose, identity, fanout, apply
)
from typeclass.runtime.core import curry

from constellations.morphisms.switches import LinearWindow, SmoothWindow
from constellations.morphisms.translate import Translate
from constellations.morphisms.matrix import Matrix
from constellations.morphisms.rotations import Rotation3D
from constellations.morphisms.sphere import Sphere
from constellations.morphisms.disk import Disk
from constellations.morphisms.boundingbox import BoundingBox
from constellations.morphisms.fit import Fit

from constellations.parsers.tree_topology import parser

from constellations.lsystems.tree_topology import lsystem

from constellations.realizations.primitives.square import square
from constellations.realizations.primitives.rectangle import rectangle

from constellations.geometry.rectangle import Rectangle
from constellations.geometry.core import SegmentStrip

from constellations.interpreters.svg import SVG

from constellations.paper.core import A0, A2, A0x2

from .utils import extract, collect_leaves, classify, sum_down_tree
from lsystems.generate import Generate

COMPOSITION_SEED = randint((2**32) - 1)

def keyed_rng(seed=0, tag=""):
    @lru_cache(maxsize=None)
    def local_seed(key):
        h = sha256((repr((seed, tag, key))).encode()).digest()
        return int.from_bytes(h[:8], "little")

    def f(key):
        return default_rng(local_seed(key))

    return f

rng_offsets  = keyed_rng(COMPOSITION_SEED, "offsets")
rng_segments = keyed_rng(COMPOSITION_SEED, "segments")
rng_widths   = keyed_rng(COMPOSITION_SEED, "widths")
rng_axes     = keyed_rng(COMPOSITION_SEED, "axes")
rng_angles   = keyed_rng(COMPOSITION_SEED, "angles")
rng_samples  = keyed_rng(COMPOSITION_SEED, "samples")

# ================================
# Tree Topology (Structure Only)
# ================================

TOPOLOGY_SEED = randint((2**32) - 1)
lsystem_result = Generate(lsystem, depth=14, seed=TOPOLOGY_SEED).run()
parsed_tree = parser.run(lsystem_result)[0][0]
pretty(parsed_tree)

# ================================
# Global Constants
# ================================

WORLD_WIDTH  = 0.6     # scales flattened XY space
WORLD_HEIGHT = 10      # scales Z axis (geometry height)

# ================================
# Domain Parameters (NOT world space)
# ================================

ALPHA = 0.7
FLOOR = 0.0
TRUNK = 0.3

height_contributions = StreamTree                                         \
    |pure| curry(lambda depth, rng:                                       \
        FLOOR + (((1.0 - FLOOR) * rng.random()) / ((1 + depth) ** ALPHA)))\
      |ap| depths()                                                       \
      |ap| (paths() |fmap| rng_segments)                                  \

node_heights = sum_down_tree(
    evaluate(height_contributions),
    TRUNK,
    is_root=True,
)

# Width parameters for activation windows
node_widths = paths()                                                     \
    |fmap| rng_widths                                                     \
    |fmap| (lambda rng: rng.random())                                     \
    |fmap| (lambda x: x / 3)


# ================================
# Rotation Control (Domain Logic)
# ================================

# Controls when rotations activate over t
rotation_windows = StreamTree                                             \
    |pure| curry(SmoothWindow)                                            \
      |ap| node_heights                                                   \
      |ap| node_widths


# Random axes on unit sphere
rotation_axes = paths()                                                   \
    |fmap| rng_axes                                                       \
    |fmap| (lambda rng: rng.random((2,)))                                 \
    |fmap| (lambda vec: 2*pi*vec)                                         \
    |fmap| Sphere()


# Rotation morphisms (axis → rotation)
rotation_morphisms = StreamTree                                           \
    |pure| curry(Rotation3D) |ap| rotation_axes                           \
    |fmap| (lambda rotate: Morphism |arrow| rotate)


# Angle scaling as function of t
angle_scalars = paths()                                                   \
    |fmap| rng_angles                                                     \
    |fmap| (lambda rng: rng.random())                                     \
    |fmap| (lambda num: 2 * pi * ((9/40) * num - (9/40)))                 \
    |fmap| (lambda angle: Morphism |arrow| (lambda t: angle * t))


# Combine: window ∘ scale ∘ rotate
# Result: t ↦ rotation morphism
local_rotations = StreamTree                                              \
    |pure| curry(lambda window, scale, rotate:
                 window |rcompose| scale |rcompose| rotate)               \
      |ap| rotation_windows                                               \
      |ap| angle_scalars                                                  \
      |ap| rotation_morphisms                                             \
    |fmap| evaluate


# ================================
# World Geometry (Tree Anchors)
# ================================

world_root = array((0.0, 0.0))

OFFSET_RADIUS = 1.0
OFFSET_ALPHA = 0.8

radii = depths()                                                          \
    |fmap| (lambda depth: OFFSET_RADIUS / (1 + depth) ** OFFSET_ALPHA)


angles = paths()                                                          \
    |fmap| rng_offsets                                                    \
    |fmap| (lambda rng: rng.random())                                     \
    |fmap| (lambda num: 2 * pi * num)


local_offsets = StreamTree                                                \
    |pure| curry(lambda radius, angle: Disk()(array([radius, angle])))    \
      |ap| radii                                                          \
      |ap| angles

world_offsets = sum_down_tree(
    evaluate(local_offsets),
    world_root,
    is_root = True
)

# Flattened XY positions (shared world frame)
world_anchor_xy = StreamTree                                              \
    |pure| curry(lambda root_, offset: root_ + offset)                    \
      |ap| (StreamTree |pure| world_root)                                 \
      |ap| world_offsets                                                  \
    |fmap| (lambda point: WORLD_WIDTH * point)


# Lift into 3D using scaled heights
world_anchor_points = StreamTree                                          \
    |pure| curry(lambda xy, z: array([*xy, z]))                           \
      |ap| world_anchor_xy                                                \
      |ap| (node_heights |fmap| (lambda h: WORLD_HEIGHT * h))


# ================================
# Local Transformations (Per Node)
# ================================

# Translate to anchor
forward_transforms = world_anchor_points |fmap| Translate

# Translate back to origin
inverse_transforms = world_anchor_points                                  \
    |fmap| ((Morphism |arrow| inverse) |compose| Translate)


# Local transform:
# t ↦ inverse → rotation(t) → forward
local_transform_functions = StreamTree                                    \
    |pure| curry(lambda inv, rot, fwd:
          Reader(lambda t: inv |rcompose| rot(t) |rcompose| fwd))         \
      |ap| inverse_transforms                                             \
      |ap| local_rotations                                                \
      |ap| forward_transforms


# ================================
# Tree Scan (Compose Downward)
# ================================

def compose_down_tree(tree, accumulated):
    local_reader = tree.value
    children = tree.children.force()

    combined_reader = Reader                                              \
        |pure| curry(lambda parent, local: local |rcompose| parent)       \
          |ap| accumulated                                                \
          |ap| local_reader                                               \
        |fmap| evaluate

    composed_children = children                                          \
        |fmap| (lambda child: compose_down_tree(child, combined_reader))

    return StreamTree(evaluate(combined_reader), interpret(composed_children))


# ================================
# Fully Composed Transformation Field
# ================================

initial_transform = Reader(lambda _: identity(Morphism))

composed_tree = compose_down_tree(
    evaluate(local_transform_functions),
    initial_transform
)


# Pair each node with its world position
tree_samples = StreamTree                                                 \
    |pure| curry(lambda pos, fn: (pos, fn))                               \
      |ap| world_anchor_xy                                                \
      |ap| composed_tree


# ================================
# Finite Sampling (Geometry)
# ================================

# Sample positions in disk → scale to world space
sample_positions = iterate(lambda i: i + 1, 0)                            \
    |fmap| rng_samples                                                    \
    |fmap| (lambda rng: rng.random((2,)))                                 \
    |fmap| (lambda vec: vec * array([1, 2*pi]))                           \
    |fmap|  Disk()                                                        \
    |fmap| (lambda xy: WORLD_WIDTH * xy)


# Canonical vertical line ("electron")
base_line = Stream                                                        \
    |pure| (Morphism |arrow| (lambda t: array([0.0,0.0,WORLD_HEIGHT*t])))


# Lift positions into 3D → translation morphisms
sample_translations = sample_positions                                    \
    |fmap| (lambda xy: array([*xy, 0]))                                   \
    |fmap| Translate


# Place line at sampled locations
placed_lines = Stream                                                     \
    |pure| curry(lambda line, loc: line |rcompose| loc)                   \
      |ap| base_line                                                      \
      |ap| sample_translations                                            \
    |fmap| evaluate


# Final finite samples: (position, geometry)
line_samples = Stream                                                     \
    |pure| curry(lambda pos, line: (pos, line))                           \
      |ap| sample_positions                                               \
      |ap| placed_lines


def machine(readerline):
    reader, line = readerline
    morphism = Morphism |arrow| reader.run
    return morphism |fanout| (Morphism, line) |rcompose| apply(Morphism)

realized_tree = extract(parsed_tree, evaluate(tree_samples))
leaves = collect_leaves(realized_tree)
machine_samples = line_samples                                            \
        |fmap| (lambda line: classify(leaves, line))                      \
        |fmap| machine                                                    \
        |fmap| evaluate


iso = array([
    [1.0, -1.0, 0.0],
    [0.5,  0.5, -1.2],
])
samples = SegmentStrip(linspace(0, 2.1, 100))
machine_samples = machine_samples                                         \
        |fmap| (lambda morphism: samples |fmap| morphism)                 \
        |fmap| (lambda segment:  segment |fmap| Matrix(iso.T))

compiled = evaluate(machine_samples)
print("compile:", time.time())

computation = evaluate(take(1000, compiled))
print("compute:", time.time())


bbox = BoundingBox()(computation)
frame = computation                                                       \
        |fmap| (lambda strip: strip |fmap| Fit(A0x2.rectangle, bbox))

SVG().write_to_file(
    f"src/constellations/compositions/signal_bloom/renders/svg/{COMPOSITION_SEED}_{TOPOLOGY_SEED}.svg"
    , evaluate(frame)
    )

print("write:", time.time())
