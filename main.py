import matplotlib.pyplot as plt
from ast import literal_eval
from acoustic_vibrations_solver import AcousticVibrationsSolver
from heat_equation_solver import HeatEquationSolver
from abstract_solver import AbstractSolver


def plot_solution(solver: AbstractSolver, color: str) -> None:
    plt.plot(solver.linspace(), solver.solve(), color=color)


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

    acoustic_vibrations_solver = AcousticVibrationsSolver(n)
    heat_equation_solver = HeatEquationSolver(n)

    # plot_solution(acoustic_vibrations_solver, "darkgreen")
    plot_solution(heat_equation_solver, "orange")

    plt.show()
