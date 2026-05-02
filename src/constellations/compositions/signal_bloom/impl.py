# ================================
# Imports
# ================================

from numpy import array, pi, linspace, diag
from numpy.random import default_rng, randint
from numpy.linalg import norm
from functools import lru_cache
from hashlib import sha256
from dataclasses import dataclass
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

from .utils import extract, collect_leaves, classify, sum_down_tree, compose_down_tree
from lsystems.generate import Generate

@dataclass(frozen=True)
class SignalBloomEnv:
    WORLD_WIDTH  : float
    WORLD_HEIGHT : float

    FLOOR : float
    ALPHA : float
    TRUNK : float

    OFFSET_RADIUS : float
    OFFSET_ALPHA  : float

    HEIGHT_SEED : int
    WIDTHS_SEED : int
    AXES_SEED   : int
    ANGLE_SEED  : int
    OFFSET_SEED : int
    SAMPLE_SEED : int

COMPOSITION_SEED = randint((2**32) - 1)

env = SignalBloomEnv(
    WORLD_WIDTH  = 0.6,
    WORLD_HEIGHT = 10,

    FLOOR = 0.0,
    ALPHA = 0.7,
    TRUNK = 0.3,

    OFFSET_RADIUS = 1.0,
    OFFSET_ALPHA  = 0.8,

    HEIGHT_SEED = randint(2**32 - 1), 
    WIDTHS_SEED = randint(2**32 - 1), 
    AXES_SEED   = randint(2**32 - 1), 
    ANGLE_SEED  = randint(2**32 - 1), 
    OFFSET_SEED = randint(2**32 - 1), 
    SAMPLE_SEED = randint(2**32 - 1), 
)

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


# ================================
# Domain Parameters (NOT world space)
# ================================

ALPHA = 0.7
FLOOR = 0.0
TRUNK = 0.3

## height_contributions :: Reader SignalBloomEnv (StreamTree Float)
height_contributions = Reader(lambda env:
    StreamTree
        |pure| curry(lambda depth, rng:
            env.FLOOR
            + ((1.0 - env.FLOOR) * rng.random())
              / ((1 + depth) ** env.ALPHA)
        )
        |ap| depths()
        |ap| (paths() |fmap| keyed_rng(env.HEIGHT_SEED, "segments"))
)


## node_heights :: Reader SignalBloomEnv (StreamTree Float)
node_heights = Reader                                                    \
    |pure| curry(lambda trunk, heights:
        sum_down_tree(
            evaluate(heights),
            trunk,
            is_root=True,
        )
    )                                                                     \
    |ap| Reader(lambda env: env.TRUNK)                                    \
    |ap| height_contributions


# Width parameters for activation windows
## node_heights :: Reader SignalBloomEnv (StreamTree Float)
node_widths = Reader(lambda env:                                          \
        paths()                                                           \
        |fmap| keyed_rng(env.WIDTHS_SEED, "widths")                       \
        |fmap| (lambda rng: rng.random())                                 \
        |fmap| (lambda x: x / 3)                                          \
    )


# ================================
# Rotation Control (Domain Logic)
# ================================

# Controls when rotations activate over t
## rotation_windows :: Reader SignalBloomEnv (StreamTree (Float -> Float))
rotation_windows = Reader                                                 \
    |pure| curry(lambda heights, widths:                                  \
        StreamTree                                                        \
            |pure| curry(SmoothWindow)                                    \
              |ap| heights                                                \
              |ap| widths                                                 \
    )                                                                     \
    |ap| node_heights                                                     \
    |ap| node_widths


# Random axes on unit sphere
## rotation_axes :: Reader SignalBloomEnv (StreamTree NDArray)
rotation_axes = Reader(lambda env:                                        \
        paths()                                                           \
        |fmap| keyed_rng(env.AXES_SEED, "axes")                           \
        |fmap| (lambda rng: rng.random((2,)))                             \
        |fmap| (lambda vec: 2*pi*vec)                                     \
        |fmap| Sphere()                                                   \
    )


# Rotation morphisms (axis → rotation)
## rotation_morphisms :: Reader SignalBloomEnv (StreamTree Morphism)
rotation_morphisms = Reader                                               \
    |pure| curry(lambda axes:                                             \
        StreamTree                                                        \
        |pure| curry(Rotation3D)                                          \
          |ap| axes                                                       \
        |fmap| (lambda rotate: Morphism |arrow| rotate)                   \
    )                                                                     \
    |ap| rotation_axes


# Angle scaling as function of t
## angle_scalars :: Reader SignalBloomEnv (StreamTree Morphism)
angle_scalars = Reader(lambda env:                                        \
        paths()                                                           \
        |fmap| keyed_rng(env.ANGLE_SEED, "angle")                         \
        |fmap| (lambda rng: rng.random())                                 \
        |fmap| (lambda num: 2 * pi * ((9/40) * num - (9/40)))             \
        |fmap| (lambda angle: Morphism |arrow| (lambda t: angle * t))     \
    )


# Combine: window ∘ scale ∘ rotate
# Result: t ↦ rotation morphism
## local_rotations :: Reader SignalBloomEnv (StreamTree Morphism)
local_rotations = Reader                                                 \
    |pure| curry(lambda windows, angles, rotations:
        StreamTree
            |pure| curry(lambda window, angle, rotate:
                window
                |rcompose| angle
                |rcompose| rotate
            )
            |ap| windows
            |ap| angles
            |ap| rotations
        |fmap| evaluate
    )                                                                     \
    |ap| rotation_windows                                                 \
    |ap| angle_scalars                                                    \
    |ap| rotation_morphisms


# ================================
# World Geometry (Tree Anchors)
# ================================

world_root = array((0.0, 0.0))

OFFSET_RADIUS = 1.0
OFFSET_ALPHA = 0.8

local_radii = Reader(lambda env:                                          \
        depths()                                                          \
        |fmap| (lambda depth: 1 + depth)                                  \
        |fmap| (lambda value: value ** env.OFFSET_ALPHA)                  \
        |fmap| (lambda value: env.OFFSET_RADIUS / value)                  \
    )


local_angles = Reader(lambda env:                                         \
        paths()                                                           \
        |fmap| keyed_rng(env.OFFSET_SEED, "offsets")                      \
        |fmap| (lambda rng: rng.random())                                 \
        |fmap| (lambda num: 2 * pi * num)                                 \
    )


local_offsets = Reader                                                   \
    |pure| curry(lambda radii, angles:
        StreamTree
            |pure| curry(lambda radius, angle:
                Disk()(array([radius, angle]))
            )
            |ap| radii
            |ap| angles
    )                                                                     \
    |ap| local_radii                                                      \
    |ap| local_angles


world_offsets = Reader                                                    \
    |pure| curry(lambda offsets:                                          \
        sum_down_tree(                                                    \
            evaluate(offsets),                                            \
            array([0.0, 0.0]),                                            \
            is_root=True,                                                 \
        )                                                                 \
    )                                                                     \
    |ap| local_offsets

# Flattened XY positions (shared world frame)
world_anchor_xy = Reader                                                  \
    |pure| curry(lambda scale, offsets: evaluate(offsets |fmap| scale))   \
    |ap| Reader(lambda env: (lambda offset: env.WORLD_WIDTH*offset))      \
    |ap| world_offsets


# Lift into 3D using scaled heights
world_anchor_points = Reader                                              \
    |pure| curry(lambda anchor_xy, heights, scale:                        \
        StreamTree                                                        \
        |pure| curry(lambda xy, z: array([*xy, z]))                       \
          |ap| anchor_xy                                                  \
          |ap| (heights |fmap| (lambda height: scale * height))           \
        )                                                                 \
      |ap| world_anchor_xy                                                \
      |ap| node_heights                                                   \
      |ap| Reader(lambda env: env.WORLD_HEIGHT)


# ================================
# Local Transformations (Per Node)
# ================================

# :: Reader SignalBloomEnv (StreamTree Morphism)
forward_transforms = Reader                                               \
    |pure| (lambda anchor_points:
        anchor_points |fmap| Translate
    )                                                                     \
    |ap| world_anchor_points


# :: Reader SignalBloomEnv (StreamTree Morphism)
inverse_transforms = Reader                                               \
    |pure| (lambda anchor_points:
        anchor_points |fmap| (lambda p: inverse(Translate(p)))
    )                                                                     \
    |ap| world_anchor_points


# :: Reader SignalBloomEnv (StreamTree Morphism)
local_transform_functions = Reader                                        \
    |pure| curry(lambda invs, rots, fwds:
        StreamTree                                                        \
            |pure| curry(lambda inv, rot, fwd:
                rot
                |rcompose| (Morphism |arrow| (lambda r:
                    inv
                    |rcompose| r
                    |rcompose| fwd
                ))
            )                                                             \
            |ap| invs                                                     \
            |ap| rots                                                     \
            |ap| fwds                                                     \
            |fmap| evaluate
    )                                                                     \
    |ap| inverse_transforms                                               \
    |ap| local_rotations                                                  \
    |ap| forward_transforms                                               \
    |fmap| evaluate


# ================================
# Fully Composed Transformation Field
# ================================


initial_transform = evaluate(identity(Morphism))

composed_tree = Reader                                                    \
    |pure| curry(lambda transform_functions:                              \
        compose_down_tree(                                                \
            evaluate(transform_functions),                                \
            initial_transform                                             \
        )                                                                 \
    )                                                                     \
    |ap| local_transform_functions


# Pair each node with its world position
tree_samples = Reader                                                     \
    |pure| curry(lambda anchor_xy, composed:                              \
        StreamTree                                                        \
        |pure| curry(lambda pos, fn: (pos, fn))                           \
          |ap| anchor_xy                                                  \
          |ap| composed                                                   \
    )                                                                     \
    |ap| world_anchor_xy                                                  \
    |ap| composed_tree                                                    \
    |fmap| evaluate


evaluation = evaluate(tree_samples).run(env)
breakpoint()

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


def machine(morphismline):
    morphsim, line = morphismline
    return morphism                                                       \
        |fanout| (Morphism, line)                                         \
        |rcompose| apply(Morphism)

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
