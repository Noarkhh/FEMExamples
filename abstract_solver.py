from abc import abstractmethod, ABC
import numpy as np
from scipy.integrate import quad
from typing import Callable
from base_function_manager import BaseFunctionManager


class AbstractSolver(ABC):
    n: int
    base: BaseFunctionManager

    def __init__(self, n: int) -> None:
        self.n = min(n, 57)
        self.base = BaseFunctionManager(self.n)

    @abstractmethod
    def B(self, u: Callable[[float], float], u_diff: Callable[[float], float],
                v: Callable[[float], float], v_diff: Callable[[float], float]) -> float: ...

    @abstractmethod
    def L(self, v: Callable[[float], float], v_diff: Callable[[float], float]) -> float: ...

    def main_matrix(self) -> np.array:
        e, e_diff = self.base.e, self.base.e_diff
        mat = np.zeros((self.n + 1, self.n + 1))
        diagonal_value = self.B(e[1], e_diff[1], e[1], e_diff[1])
        diagonal_neighbour_value = max(self.B(e[1], e_diff[1], e[2], e_diff[2]),
                                       self.B(e[2], e_diff[2], e[1], e_diff[1]))
        for i in range(1, self.n):
            mat[i - 1, i - 1] = diagonal_value
            mat[i - 1, i] = mat[i, i - 1] = diagonal_neighbour_value

        mat[self.n - 1, self.n - 1] = self.B(e[self.n], e_diff[self.n], e[self.n], e_diff[self.n])
        return mat

    def coefficient_vector(self) -> np.array:
        vect = np.zeros(self.n)
        for j in range(self.n):
            vect[j] = self.L(self.base.e[j + 1], self.base.e_diff[j + 1])
        return vect

    def solve(self) -> list[float]:
        result = np.linalg.solve(self.main_matrix(), self.coefficient_vector())
        return [2] + [w + 2 for w in result]
