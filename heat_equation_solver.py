from abstract_solver import AbstractSolver
from scipy.integrate import quad
from typing import Callable


class HeatEquationSolver(AbstractSolver):
    def __init__(self, n):
        super().__init__(n, 0, 2, right_boundary_zeroed=True)

    def B(self, u: Callable[[float], float], u_diff: Callable[[float], float],
                v: Callable[[float], float], v_diff: Callable[[float], float]) -> float:
        return (u(0) * v(0) +
                -quad(lambda x: u_diff(x) * v_diff(x), 0, 2)[0])

    def L(self, v: Callable[[float], float], v_diff: Callable[[float], float]) -> float:
        return (-quad(lambda x: (100 * x * v(x)) / (x + 1), 0, 1)[0] +
                -quad(lambda x: v(x), 1, 2)[0] * 50 +
                20 * v(0))

