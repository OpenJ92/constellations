from typeclass.data.tree import Tree, pretty
from typeclass.data.sequence import Sequence
from typeclass.data.parser import Parser, char, none_of
from typeclass.interpret.run import evaluate
from typeclass.typeclasses.symbols import many, then, skip, fmap, otherwise

def delay(f):
    return Parser(lambda s: evaluate(f()).run(s))

def junk():
    return Parser |many| none_of("[]")

def token(c):
    return junk() |then| char(c)

def _tree():
    return tree

lbrack, rbrack = token("["), token("]")

leaf   = lbrack |then| rbrack
branch = lbrack |then| (Parser |many| delay(_tree)) |skip| rbrack

leaf   = leaf   |fmap| (lambda _: Tree(None, Sequence([])))
branch = branch |fmap| (lambda xs: Tree(None, Sequence(xs)))
tree   = leaf |otherwise| branch

parser = evaluate(tree)

