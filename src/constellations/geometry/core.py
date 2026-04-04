from typeclass.data.sequence import Sequence


class SegmentStrip(Sequence):
    def fmap(self, f):
        _f = f.force()
        return SegmentStrip(tuple(_f(x) for x in self._values))

    def __repr__(self) -> str:
        return f"SegmentStrip({self._values!r})"
