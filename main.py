import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from abstract_solver import Solver


if __name__ == "__main__":
    n: int = 10
    while True:
        n_str = input("n: int = ")
        try:
            n = literal_eval(n_str)
            if not isinstance(n, int) or n < 2:
                raise ValueError
        except (SyntaxError, ValueError):
            print("Illegal value of n!")
            continue
        break

    solver = Solver(n)
    plt.plot(np.linspace(0, 2, solver.n + 1), solver.solve(), color="darkgreen")
    plt.show()
