from typeclass.data.sequence import Sequence, zipwith
from typeclass.data.stream import Stream, take, iterate
from typeclass.data.tree import Tree
from typeclass.data.reader import Reader
from typeclass.data.streamtree import paths
from typeclass.typeclasses.symbols import fmap, pure, ap, rcompose
from typeclass.runtime.core import curry

from functools import lru_cache
from hashlib import sha256
from numpy import array, pi, exp, cos, sin
from numpy.linalg import norm
from numpy.random import default_rng

from typeclass.data.streamtree import StreamTree
from typeclass.typeclasses.symbols import fmap
from typeclass.interpret.run import evaluate, interpret

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


def sum_down_tree(tree, accumulated, is_root=False):
    local = tree.value
    children = tree.children.force()

    combined = accumulated if is_root else accumulated + local

    summed_children = children \
        |fmap| (lambda child: sum_down_tree(child, combined))

    return StreamTree(combined, interpret(summed_children))


def compose_down_tree(tree, accumulated):
    local = tree.value
    children = tree.children.force()

    combined = evaluate(local |rcompose| accumulated)

    composed_children = children                                          \
        |fmap| (lambda child: compose_down_tree(child, combined))

    return StreamTree(combined, interpret(composed_children))


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
