from lsystems.sentences.string import String
from lsystems.productions.static import Static
from lsystems.productions.stochastic import Stochastic
from lsystems.productions.productions import Productions
from lsystems.lsystem import LSystem


sentence = String("[X]")

productions = Productions(String)

stochastic = Stochastic()
stochastic.add(10, Static(String("[X]")))
stochastic.add(10, Static(String("[[X]]")))
stochastic.add(10, Static(String("X[X]")))
stochastic.add(10, Static(String("[X]X")))
stochastic.add(15, Static(String("[X][X][X]")))
stochastic.add(10, Static(String("[X][X]")))
stochastic.add(10, Static(String("X[X]X")))
stochastic.add(10, Static(String("[X[X]]")))
stochastic.add(10, Static(String("[X][[X]]")))

productions.add("X", stochastic)

alphabet = set("X[]")

lsystem = LSystem(alphabet, productions, sentence)


