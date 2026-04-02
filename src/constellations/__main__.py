from constellations.morphisms.partitionby.line import Line2D, PartitionByLine
from constellations.geometry.core import SegmentStrip
from numpy import array, linspace

from typeclass.data.sequence import Sequence, concat
from typeclass.data.morphism import Morphism
from typeclass.typeclasses.symbols import fmap, rcompose, arrow
from typeclass.interpret.run import evaluate

cross = Sequence([SegmentStrip([array([value+k, value]) for value in linspace(-20, 20, 100)]) for k in range(10)])

vertical = Line2D(array([0, 1]), array([0, 0]))
vpartition = PartitionByLine(vertical)

horizontal = Line2D(array([1, 0]), array([0, 0]))
hpartition = PartitionByLine(horizontal)

def sequence_tuple2(xs):
    left = []
    right = []

    for a, b in xs:
        left.append(a)
        right.append(b)

    return Sequence(left), Sequence(right)

def concat(xss):
    out = []
    for xs in xss:
        for x in xs:
            out.append(x)
    return Sequence(out)

def bimap(f, g):
    return lambda pair: (f(pair[0]), g(pair[1]))


lift_partition = lambda f: (
    (Morphism |arrow| (lambda xs: xs |fmap| f))
    |rcompose| (Morphism |arrow| sequence_tuple2)
    |rcompose| (Morphism |arrow| bimap(concat, concat))
)

split_x = evaluate(lift_partition(vpartition))
split_y = evaluate(lift_partition(hpartition))

left, right = split_x(cross)

ul, ll = split_y(left)
ur, lr = split_y(right)



