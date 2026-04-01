from typeclass.data.morphism import Morphism

class Matrix(Morphism):
    def __init__(self, matrix):
        self.matrix = matrix

    def _run(self, x):
        return self.matrix.T @ x
