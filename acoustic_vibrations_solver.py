from abstract_solver import AbstractSolver
from scipy.integrate import quad
from typing import Callable


class AcousticVibrationsSolver(AbstractSolver):
    def B(self, u: Callable[[float], float], u_diff: Callable[[float], float],
                v: Callable[[float], float], v_diff: Callable[[float], float]) -> float:
        return ((u(2) * v(2)) +
                quad(lambda x: u_diff(x) * v_diff(x), 0, 2)[0] -
                quad(lambda x: u(x) * v(x), 0, 2)[0])

    def L(self, v: Callable[[float], float], v_diff: Callable[[float], float]) -> float:
        return (quad(lambda x: v(x) * np.sin(x), 0, 2)[0] -
                self.B(lambda x: 2, lambda x: 0, v, v_diff))
