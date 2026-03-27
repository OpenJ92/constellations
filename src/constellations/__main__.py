from numpy import array, pi
from numpy.random import random

from typeclass.data.stream import Stream, take
from typeclass.data.automorphism import Automorphism
from typeclass.data.parser import Parser, char, none_of
from typeclass.data.parser.lib import delay
from typeclass.data.thunk import suspend
from typeclass.data.tree import Tree, pretty
from typeclass.data.sequence import Sequence
from typeclass.interpret.run import run, evaluate
from typeclass.typeclasses.symbols import fmap, pure, ap, compose, many, then, skip, otherwise
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

def delay(f):
    return Parser(lambda s: evaluate(f()).run(s))

def junk():
    return Parser |many| none_of("[]")

def token(c):
    return junk() |then| char(c)

def tree_parser():
    tree = None

    def _tree():
        return tree

    lbrack = token("[")
    rbrack = token("]")

    leaf   = lbrack |then| rbrack
    branch = lbrack |then| (Parser |many| delay(_tree)) |skip| rbrack

    leaf   = leaf   |fmap| (lambda _: Tree(None, Sequence([])))
    branch = branch |fmap| (lambda xs: Tree(None, Sequence(xs)))
    tree = leaf |otherwise| branch

    return tree

parser = evaluate(tree_parser())

def extract(shape, st):
    children = shape.children
    n = length(children)
    st_children = take(n, st.children.force())

    return Tree(
        st.value,
        zip_with(extract, children, st_children)
    )

# Start symbol
sentence = String("X")

# Productions (deterministic)
productions = Productions(String)
productions.add("X", Static(String("F+[[X]-X]-F[-FX]+X")))
productions.add("F", Static(String("FF")))

# Alphabet includes every symbol we might emit
alphabet = set("FX+-[]")

lsys = LSystem(alphabet, productions, sentence)

gen = Generate(lsys, depth=6)
result = gen.run()

results = parser.run(result)
tree = evaluate(results[0][0] |fmap| (lambda _: random()))
