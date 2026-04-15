from typeclass.data.sequence import Sequence, zipwith
from typeclass.data.stream import Stream, take, iterate
from typeclass.data.tree import Tree
from typeclass.data.reader import Reader
from typeclass.data.streamtree import paths
from typeclass.typeclasses.symbols import fmap, pure, ap
from typeclass.runtime.core import curry

from functools import lru_cache
from hashlib import sha256
from numpy import array, pi, exp, cos, sin
from numpy.random import default_rng

def length(sequence):
    return len(sequence._values)

def extract(shape, st):
    children = shape.children
    n = length(children)
    st_children = take(n, st.children.force())

    return Tree(
        st.value,
        Sequence(tuple(zipwith(extract, children, st_children)))
    )

def path_random(seed=0):
    def f(p):
        h = sha256((str(p) + str(seed)).encode()).digest()
        rng = default_rng(int.from_bytes(h[:8], "little"))
        return rng.random()
    return f

def ascending_segment_length_tree(alpha: float = 0.7, floor: float = 0.0):
    r = path_random()

    return paths() |fmap| (
            lambda p: sum(
                floor + (1.0 - floor) * r(p[:i]) / (1 + i)**alpha
                for i in range(1, len(p) + 1)
            )
        )

def radius_for_depth( depth, base_radius=1.0, alpha= 0.8,):
    return base_radius / (1 + depth) ** alpha


def angle_for_prefix(prefix, angle_seed=0,):
    r = path_random(angle_seed)
    return 2 * pi * r(prefix)


def offset_for_prefix(prefix, base_radius=1.0, alpha=0.8, angle_seed=0,):
    depth = len(prefix) - 1
    radius = radius_for_depth(depth, base_radius, alpha,)
    angle = angle_for_prefix(prefix, angle_seed)

    return array(( radius * cos(angle), radius * sin(angle),))


def make_offset_from_path(base_radius, alpha=0, angle_seed=3):
    @lru_cache(maxsize=None)
    def offset_from_path(p):
        if len(p) == 0:
            return array((0.0, 0.0))

        parent = p[:-1]
        return (offset_from_path(parent) + offset_for_prefix(p, base_radius, alpha, angle_seed,))

    return offset_from_path


def make_reader_composition_from_path():
    @lru_cache(maxsize=None)
    def reader_composition_from_path(p):
        if len(p[0]) == 0:
            return Reader(lambda x: x)

        path, reader = p
        parent = path[:-1]


        return Reader\
          |pure| curry(lambda r, q: r |compose| q)\
            |ap| reader_composition_from_path(parent)\
            |ap| reader

    return reader_composition_from_path
