from numpy import array, pi
from numpy.random import random

from typeclass.data.stream import Stream, take
from typeclass.data.automorphism import Automorphism
from typeclass.data.parser import Parser, char
from typeclass.data.parser.lib import delay
from typeclass.data.thunk import suspend
from typeclass.data.tree import Tree, pretty
from typeclass.data.sequence import Sequence
from typeclass.interpret.run import run, evaluate
from typeclass.typeclasses.symbols import fmap, pure, ap, compose, many, then, skip, otherwise
from typeclass.runtime.core import curry

widths  = Stream |pure| None |fmap| (lambda _: random())
heights = Stream |pure| None |fmap| (lambda _: random()*2*pi)

class Translate(Automorphism):
    def __init__(self, vector):
        self.vector = vector

    def _run(self, x):
        return self.vector + x

    def _inv(self):
        return -1*self.vector


vectors = Stream |pure| curry(lambda x, y: array((x, y))) |ap| widths |ap| heights |fmap| Translate
wectors = Stream |pure| curry(lambda x, y: array((x, y))) |ap| widths |ap| heights |fmap| Translate
composed = Stream |pure| curry(lambda x, y: evaluate(x |compose| y)) |ap| vectors |ap| wectors

def sep_by1(p, sep):
    return (
        (Parser |pure| (lambda x: lambda xs: [x] + xs))
        |ap| p
        |ap| (Parser |many| (sep |then| p))
    )

def sep_by(p, sep):
    return sep_by1(p, sep) |otherwise| (Parser |pure| [])

def delay(f):
    return Parser(lambda s: evaluate(f()).run(s))

def tree_parser():
    tree = None

    def delayed_tree():
        return tree

    leaf = (char("[") |then| char("]"))
    children = sep_by(delay(delayed_tree), char(","))
    branch = (char("[") |then| children |skip| char("]"))

    leaf   =   leaf |fmap| (lambda _: Tree(None, Sequence([])))
    branch = branch |fmap| (lambda xs: Tree(None, Sequence(xs)))

    tree = leaf |otherwise| branch
    return tree

parser = evaluate(tree_parser())
results = parser.run("[[[[[[]],[]],[[[],[]],[]]],[[[],[[]]],[[],[]]]],[[[],[]],\
                      [[[[]],[]],[[],[[],[]]]]],[],[[[[],[]],[[[]],[]]],[[[],[[\
                      ],[]]],[]]],[[],[[[],[]],[[[]],[[],[]]]]],[[[[[]]],[]],[[\
                      ],[[],[[[]],[]]]]]]")

tree = evaluate(results[0][0] |fmap| (lambda _: random()))

