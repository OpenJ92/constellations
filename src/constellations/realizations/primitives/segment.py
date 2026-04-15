from numpy import linspace
from typeclass.data.reader import Reader

from constellations.geometry.core import SegmentStrip

segment = Reader(lambda s: SegmentStrip(linspace(0, 1, s, endpoint=False)))
