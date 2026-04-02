from typeclass.data.sequence import Sequence


class SegmentStrip(Sequence):
    def __repr__(self) -> str:
        return f"SegmentStrip({self._values!r})"
