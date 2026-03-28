from hashlib import sha256
from numpy import array, pi
from numpy.random import random, default_rng

from typeclass.data.stream import Stream, take
from typeclass.data.streamtree import StreamTree, paths, coordinates, depths
from typeclass.data.automorphism import Automorphism
from typeclass.data.parser import Parser, char, none_of
from typeclass.data.parser.lib import delay
from typeclass.data.tree import Tree, pretty
from typeclass.data.sequence import Sequence, zipwith
from typeclass.data.maybe import Just
from typeclass.data.thunk import suspend
from typeclass.interpret.run import run, evaluate
from typeclass.typeclasses.symbols import fmap, pure, ap, compose, many, then, skip, otherwise, bind
from typeclass.runtime.core import curry

from lsystems.sentences.string import String
from lsystems.sentences.tuple import Tuple
from lsystems.productions.static import Static
from lsystems.productions.stochastic import Stochastic
from lsystems.productions.precedence import Precedence
from lsystems.productions.context import ContextSensitive, VariationalContextSensitive
from lsystems.productions.generation import Generation
from lsystems.productions.productions import Productions
from lsystems.lsystem import LSystem
from lsystems.generate import Generate


# ============================================================
# Morphism Generation over Stream
# ============================================================

widths  = Stream |pure| None |fmap| (lambda _: random())
heights = Stream |pure| None |fmap| (lambda _: random()*2*pi)

class Translate(Automorphism):
    def __init__(self, vector):
        self.vector = vector

    def _run(self, x):
        return self.vector + x

    def _inv(self):
        return -1*self.vector


vectors  = Stream |pure| curry(lambda x, y: array((x, y))) |ap| widths |ap| heights |fmap| Translate
wectors  = Stream |pure| curry(lambda x, y: array((x, y))) |ap| widths |ap| heights |fmap| Translate
composed = Stream |pure| curry(lambda x, y: evaluate(x |compose| y)) |ap| vectors |ap| wectors

# ============================================================
# L-System tree topology Parser 
# ============================================================

def delay(f):
    return Parser(lambda s: evaluate(f()).run(s))

def junk():
    return Parser |many| none_of("[]")

def token(c):
    return junk() |then| char(c)


def _tree():
    return tree

lbrack = token("[")
rbrack = token("]")

leaf   = lbrack |then| rbrack
branch = lbrack |then| (Parser |many| delay(_tree)) |skip| rbrack

leaf   = leaf   |fmap| (lambda _: Tree(None, Sequence([])))
branch = branch |fmap| (lambda xs: Tree(None, Sequence(xs)))
tree   = leaf |otherwise| branch
tree   = leaf |otherwise| branch

parser = evaluate(tree)

# ============================================================
# Tree topology L-System Spec
# ============================================================

def length(sequence):
    return len(sequence._values)

def extract(shape, st):
    children = shape.children
    n = length(children)
    st_children = take(n, st.children.force())

    return Tree(
        st.value,
        Sequence(zipwith(extract, children, st_children))
    )

# Start symbol
sentence = String("[X]")

# Productions (deterministic)
productions = Productions(String)

stochastic = Stochastic()
stochastic.add(10, Static(String("[X]")))
stochastic.add(10, Static(String("X[X]")))
stochastic.add(10, Static(String("[X]X")))
stochastic.add(10, Static(String("[X][X]")))
stochastic.add(10, Static(String("X[X]X")))
stochastic.add(10, Static(String("[X[X]]")))
stochastic.add(10, Static(String("[X][[X]]")))

productions.add("X", stochastic)

alphabet = set("X[]")
lsys = LSystem(alphabet, productions, sentence)
gen = Generate(lsys, depth=5)

result = gen.run()

# ============================================================
# Morphism generation over StreamTree and extraction
# ============================================================

tree = parser.run(result)[0][0]
stree = evaluate(paths())
extracted = extract(tree, stree)

def path_random(seed=0):
    def f(p):
        h = sha256((str(p) + str(seed)).encode()).digest()
        rng = default_rng(int.from_bytes(h[:8], "little"))
        return rng.random()
    return f

def ascending_random_tree():
    r = path_random()

    return evaluate(
        paths() |fmap| (
            lambda p: sum(r(p[:i]) for i in range(len(p) + 1))
        )
    )
extracted_ = extract(tree, ascending_random_tree())

# ============================================================
# Lambda opacity issue with linked |bind| operations. Post 
# recursion on Bind case suspend eval seems to fix problem
# ============================================================
expression = Just(10)                      \
    |bind| evaluate(lambda x: Just(x + 10) \
    |bind| evaluate(lambda y: Just(y + x)  \
    |bind| evaluate(lambda z: (x, y, z))))

