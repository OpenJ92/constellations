"""
Constellations — Main Construction File

Overview
--------

This file builds a *transformation field* over a tree structure and applies it
to sampled finite geometry.

Pipeline:

1. Build a tree of local, parameterized (t-dependent) transformations
2. Compose those transformations from root → leaves (top-down accumulation)
3. Sample finite geometry in a shared world coordinate system
4. Prepare (position, geometry) pairs for later classification + application

Key Idea:
- Tree side = structured transformation field
- Stream side = finite geometry samples
- Final step (not here) = match samples → transformations → apply

Terminology:
- "Electron" = the single base object to be replicated
- "Constellation" = the final set of transformed objects
"""


# ================================
# Imports
# ================================

from numpy import array, pi, linspace
from numpy.random import random, seed
from numpy.linalg import norm
import time

from typeclass.data.sequence import Sequence
from typeclass.data.stream import Stream, take
from typeclass.data.streamtree import StreamTree, paths
from typeclass.data.morphism import Morphism
from typeclass.data.reader import Reader
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

from constellations.parsers.tree_topology import parser
from constellations.lsystems.tree_topology import lsystem

from .utils import ascending_segment_length_tree, make_offset_from_path, extract
from lsystems.generate import Generate

seed(134374)

print("START:", time.time())

# ================================
# Tree Topology (Structure Only)
# ================================
lsystem_result = Generate(lsystem, depth=11).run()
print("lsystem:", time.time())
parsed_tree = parser.run(lsystem_result)[0][0]
print("parser:", time.time())


# ================================
# Global Constants
# ================================

WORLD_WIDTH  = 0.6     # scales flattened XY space
WORLD_HEIGHT = 40      # scales Z axis (geometry height)


# ================================
# Domain Parameters (NOT world space)
# ================================

# Tree-based heights (used for domain control, not geometry directly)
node_heights = ascending_segment_length_tree()                            \
    |fmap| (lambda t: t + .3)

# Width parameters for activation windows
node_widths = StreamTree                                                  \
    |pure| None                                                           \
    |fmap| (lambda _: random())                                           \
    |fmap| (lambda x: x / 6)


# ================================
# Rotation Control (Domain Logic)
# ================================

# Controls when rotations activate over t
rotation_windows = StreamTree                                             \
    |pure| curry(SmoothWindow)                                            \
      |ap| node_heights                                                   \
      |ap| node_widths


# Random axes on unit sphere
rotation_axes = StreamTree                                                \
    |pure| None                                                           \
    |fmap| (lambda _: Sphere()(2*pi*random((2,))))


# Rotation morphisms (axis → rotation)
rotation_morphisms = StreamTree                                           \
    |pure| curry(Rotation3D) |ap| rotation_axes                           \
    |fmap| (lambda rotate: Morphism |arrow| rotate)


# Angle scaling as function of t
angle_scalars = StreamTree                                                \
    |pure| None                                                           \
    |fmap| (lambda _: 2 * pi * ((1/4) * random() - (1/4)))              \
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

path_to_offset = make_offset_from_path(1.0, 0.8, 3)


# Flattened XY positions (shared world frame)
world_anchor_xy = StreamTree                                              \
    |pure| curry(lambda root_, offset: root_ + offset)                    \
      |ap| (StreamTree |pure| world_root)                                 \
      |ap| (paths() |fmap| path_to_offset)                                \
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
    """
    Top-down accumulation of transformations.

    At each node:
        new = parent ∘ local

    Produces:
        StreamTree of fully composed transformations
    """
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
sample_positions = Stream                                                 \
    |pure| None                                                           \
    |fmap| (lambda _: random((2,)) * array([1, 2*pi]))                    \
    |fmap| Disk()                                                         \
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

def collect_leaves(tree):
    if not tree.children._values:
        return [tree.value]
    out = []
    for child in tree.children._values:
        out.extend(collect_leaves(child))
    return out

def classify(leaves, line_sample):
    lposition, line = line_sample
    pos, reader = min(
        leaves,
        key=lambda leaf: norm(lposition - leaf[0]),
    )
    return reader, line

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

