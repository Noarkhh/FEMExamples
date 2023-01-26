from typing import Callable


class BaseFunctionManager:
    n: int
    h: float
    e: list[Callable[[float], float]]
    e_diff: list[Callable[[float], float]]

    def __init__(self, n):
        self.n = n
        self.h = 2/n
        self.e = [self.create_base_function(i) for i in range(n + 1)]
        self.e_diff = [self.create_base_function_derivative(i) for i in range(n + 1)]

    def create_base_function(self, i: int) -> Callable[[float], float]:
        h = self.h
        return (lambda x: (x / h - i + 1) if (h * (i - 1)) <= x < (h * i) else
                          (-x / h + i + 1) if (h * i) <= x <= (h * (i + 1)) else
                          0)

    def create_base_function_derivative(self, i: int) -> Callable[[float], float]:
        h = self.h
        return (lambda x: (1 / h) if (h * (i - 1)) <= x < (h * i) else
                          (-1 / h) if (h * i) <= x <= (h * (i + 1)) else
                          0)
