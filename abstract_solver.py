from abc import abstractmethod, ABC
import numpy as np
from typing import Callable
from base_function_manager import BaseFunctionManager


class AbstractSolver(ABC):
    n: int
    shift: float
    left_bound: float
    right_bound: float
    left_boundary_zeroed: bool
    right_boundary_zeroed: bool

    base: BaseFunctionManager

    def __init__(self, n: int, left_bound: float, right_bound: float, left_boundary_zeroed: bool = False,
                 right_boundary_zeroed: bool = False, shift: float = 0) -> None:
        self.n = min(n, 57)
        self.shift = shift
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.left_boundary_zeroed = left_boundary_zeroed
        self.right_boundary_zeroed = right_boundary_zeroed

        self.base = BaseFunctionManager(self.n, right_bound - left_bound)

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
        for i in range(self.n):
            mat[i, i] = diagonal_value
            mat[i, i + 1] = mat[i + 1, i] = diagonal_neighbour_value

        mat[self.n, self.n] = self.B(e[self.n], e_diff[self.n], e[self.n], e_diff[self.n])
        mat[0, 0] = self.B(e[0], e_diff[0], e[0], e_diff[0])

        if self.left_boundary_zeroed:
            mat[0, 0] = 1
            mat[0, 1] = mat[1, 0] = 0
        if self.right_boundary_zeroed:
            mat[self.n, self.n] = 1
            mat[self.n - 1, self.n] = mat[self.n, self.n - 1] = 0

        return mat

    def coefficient_vector(self) -> np.array:
        vect = np.zeros(self.n + 1)
        for j in range(self.n + 1):
            vect[j] = self.L(self.base.e[j], self.base.e_diff[j])
        if self.left_boundary_zeroed:
            vect[0] = 0
        if self.right_boundary_zeroed:
            vect[self.n] = 0

        return vect

    def solve(self) -> np.array:
        return np.linalg.solve(self.main_matrix(), self.coefficient_vector()) + self.shift

    def linspace(self) -> np.array:
        return np.linspace(self.left_bound, self.right_bound, self.n + 1)
