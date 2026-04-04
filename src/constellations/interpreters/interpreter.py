from typing import Protocol, runtime_checkable, Any


@runtime_checkable
class Interpreter(Protocol):
    def run(self, data: Any) -> str: ...
