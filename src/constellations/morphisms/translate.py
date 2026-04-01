from typeclass.data.automorphism import Automorphism

class Translate(Automorphism):
    def __init__(self, vector):
        self.vector = vector

    def _run(self, x):
        return self.vector + x

    def _inv(self, x):
        return Translate(-1*self.vector)(x)
