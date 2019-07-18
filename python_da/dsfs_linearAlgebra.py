from typing import List

Vector = List[float]

def add_vectors(x: Vector, y: Vector) -> Vector:
    assert len(x) == len(y), "Vectors must have same length."
    return [x_i + y_i for x_i, y_i in zip(x, y)]

assert add_vectors([-1, 2], [2, -1]) == [1, 1], "Something went wrong"

def substract_vectors(x: Vector, y: Vector) -> Vector:
    assert len(x) == len(y), "Vectors must have same length."
    return [x_i - y_i for x_i, y_i in zip(x, y)]

assert substract_vectors([2, 2], [3, 3]) == [-1, -1], "Something went wrong"

