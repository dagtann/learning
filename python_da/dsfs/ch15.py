# Multiple Regression
from scratch.linear_algebra import dot, Vector
from typing import List
import random
import tqdm
from scratch.linear_algebra import vector_mean
from scratch.gradient_descent import gradient_step
from scratch.statistics import daily_minutes_good

inputs: List[List[float]] = [
    [1., 49, 4, 0], [1, 41, 9, 0], [1, 40, 8, 0], [1, 25, 6, 0],
    [1, 21, 1, 0], [1, 21, 0, 0], [
        1, 19, 3, 0], [1, 19, 0, 0],
    [1, 18, 9, 0], [1, 18, 8, 0], [
        1, 16, 4, 0], [1, 15, 3, 0],
    [1, 15, 0, 0], [1, 15, 2, 0], [
        1, 15, 7, 0], [1, 14, 0, 0],
    [1, 14, 1, 0], [1, 13, 1, 0], [
        1, 13, 7, 0], [1, 13, 4, 0],
    [1, 13, 2, 0], [1, 12, 5, 0], [
        1, 12, 0, 0], [1, 11, 9, 0],
    [1, 10, 9, 0], [1, 10, 1, 0], [
        1, 10, 1, 0], [1, 10, 7, 0],
    [1, 10, 9, 0], [1, 10, 1, 0], [
        1, 10, 6, 0], [1, 10, 6, 0],
    [1, 10, 8, 0], [1, 10, 10, 0], [
        1, 10, 6, 0], [1, 10, 0, 0],
    [1, 10, 5, 0], [1, 10, 3, 0], [
        1, 10, 4, 0], [1, 9, 9, 0],
    [1, 9, 9, 0], [1, 9, 0, 0], [
        1, 9, 0, 0], [1, 9, 6, 0],
    [1, 9, 10, 0], [1, 9, 8, 0], [
        1, 9, 5, 0], [1, 9, 2, 0],
    [1, 9, 9, 0], [1, 9, 10, 0], [
        1, 9, 7, 0], [1, 9, 2, 0],
    [1, 9, 0, 0], [1, 9, 4, 0], [
        1, 9, 6, 0], [1, 9, 4, 0],
    [1, 9, 7, 0], [1, 8, 3, 0], [
        1, 8, 2, 0], [1, 8, 4, 0],
    [1, 8, 9, 0],
    [1, 8, 2, 0],
    [1, 8, 3, 0],
    [1, 8, 5, 0],
    [1, 8, 8, 0],
    [1, 8, 0, 0],
    [1, 8, 9, 0],
    [1, 8, 10, 0],
    [1, 8, 5, 0],
    [1, 8, 5, 0],
    [1, 7, 5, 0],
    [1, 7, 5, 0],
    [1, 7, 0, 0],
    [1, 7, 2, 0],
    [1, 7, 8, 0],
    [1, 7, 10, 0],
    [1, 7, 5, 0],
    [1, 7, 3, 0],
    [1, 7, 3, 0],
    [1, 7, 6, 0],
    [1, 7, 7, 0],
    [1, 7, 7, 0],
    [1, 7, 9, 0],
    [1, 7, 3, 0],
    [1, 7, 8, 0],
    [1, 6, 4, 0],
    [1, 6, 6, 0],
    [1, 6, 4, 0],
    [1, 6, 9, 0],
    [1, 6, 0, 0],
    [1, 6, 1, 0],
    [1, 6, 4, 0],
    [1, 6, 1, 0],
    [1, 6, 0, 0],
    [1, 6, 7, 0],
    [1, 6, 0, 0],
    [1, 6, 8, 0],
    [1, 6, 4, 0],
    [1, 6, 2, 1],
    [1, 6, 1, 1],
    [1, 6, 3, 1],
    [1, 6, 6, 1],
    [1, 6, 4, 1],
    [1, 6, 4, 1],
    [1, 6, 1, 1],
    [1, 6, 3, 1],
    [1, 6, 4, 1],
    [1, 5, 1, 1],
    [1, 5, 9, 1],
    [1, 5, 4, 1],
    [1, 5, 6, 1],
    [1, 5, 4, 1],
    [1, 5, 4, 1],
    [1, 5, 10, 1],
    [1, 5, 5, 1],
    [1, 5, 2, 1],
    [1, 5, 4, 1],
    [1, 5, 4, 1],
    [1, 5, 9, 1],
    [1, 5, 3, 1],
    [1, 5, 10, 1],
    [1, 5, 2, 1],
    [1, 5, 2, 1],
    [1, 5, 9, 1],
    [1, 4, 8, 1],
    [1, 4, 6, 1],
    [1, 4, 0, 1],
    [1, 4, 10, 1],
    [1, 4, 5, 1],
    [1, 4, 10, 1],
    [1, 4, 9, 1],
    [1, 4, 1, 1],
    [1, 4, 4, 1],
    [1, 4, 4, 1],
    [1, 4, 0, 1],
    [1, 4, 3, 1],
    [1, 4, 1, 1],
    [1, 4, 3, 1],
    [1, 4, 2, 1],
    [1, 4, 4, 1],
    [1, 4, 4, 1],
    [1, 4, 8, 1],
    [1, 4, 2, 1],
    [1, 4, 4, 1],
    [1, 3, 2, 1],
    [1, 3, 6, 1],
    [1, 3, 4, 1],
    [1, 3, 7, 1],
    [1, 3, 4, 1],
    [1, 3, 1, 1],
    [1, 3, 10, 1],
    [1, 3, 3, 1],
    [1, 3, 4, 1],
    [1, 3, 7, 1],
    [1, 3, 5, 1],
    [1, 3, 6, 1],
    [1, 3, 1, 1],
    [1, 3, 6, 1],
    [1, 3, 10, 1],
    [1, 3, 2, 1],
    [1, 3, 4, 1],
    [1, 3, 2, 1],
    [1, 3, 1, 1],
    [1, 3, 5, 1],
    [1, 2, 4, 1],
    [1, 2, 2, 1],
    [1, 2, 8, 1],
    [1, 2, 3, 1],
    [1, 2, 1, 1],
    [1, 2, 9, 1],
    [1, 2, 10, 1],
    [1, 2, 9, 1],
    [1, 2, 4, 1],
    [1, 2, 5, 1],
    [1, 2, 0, 1],
    [1, 2, 9, 1],
    [1, 2, 9, 1],
    [1, 2, 0, 1],
    [1, 2, 1, 1],
    [1, 2, 1, 1],
    [1, 2, 4, 1],
    [1, 1, 0, 1],
    [1, 1, 2, 1],
    [1, 1, 2, 1],
    [1, 1, 5, 1],
    [1, 1, 3, 1],
    [1, 1, 10, 1],
    [1, 1, 6, 1],
    [1, 1, 0, 1],
    [1, 1, 8, 1],
    [1, 1, 6, 1],
    [1, 1, 4, 1],
    [1, 1, 9, 1],
    [1, 1, 9, 1],
    [1, 1, 4, 1],
    [1, 1, 2, 1],
    [1, 1, 9, 1],
    [1, 1, 0, 1],
    [1, 1, 8, 1],
    [1, 1, 6, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 5, 1]]


def predict(x: Vector, beta: Vector) -> float:  # multiple regression model xb
    """Assumes the first element in x is a column of 1s"""
    return dot(x, beta)  # a.k.a. linear predictor


def error(x: Vector, y: float, beta: Vector) -> float:
    return predict(x, beta) - y


def squared_error(x: Vector, y: float, beta: Vector) -> float:
    return error(x, y, beta) ** 2


# test code
x = [1, 2, 3]
y = 30
beta = [4, 4, 4]  # yhat = 1 * 4 + 2 * 4 + 3 * 4

assert predict(x, beta) == 24
assert error(x, y, beta) == -6
assert squared_error(x, y, beta) == 36


def sqerror_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]


assert sqerror_gradient(x, y, beta) == [-12, -24, -36]


def least_squares_fit(xs: List[Vector],
                      ys: List[float],
                      learning_rate: float = 0.001, num_steps: int = 1000,
                      batch_size: int = 1) -> Vector:
    """
    Find the beta that minimizes the sum of squared errors
    assuming the model y = dot(x, beta)
    """

    # Start with a random guess
    guess = [random.random() for _ in xs[0]]

    for _ in tqdm.trange(num_steps, desc="least squares fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start + batch_size]
            batch_ys = ys[start:start + batch_size]

            gradient = vector_mean([sqerror_gradient(x, y, guess)
                                    for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate)

    return guess


random.seed(0)
learning_rate = 0.001

beta = least_squares_fit(inputs, daily_minutes_good, learning_rate, 5000, 25)
beta


# Goodness of fit
from scratch.simple_linear_regression import total_sum_of_squares


def multiple_r_squared(xs: List[Vector], ys: Vector, beta: Vector) -> float:
    sum_of_squared_errors = sum(error(x, y, beta) ** 2
                                for x, y in zip(xs, ys))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(ys)


assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta) < 0.68

# Bootstrap data
from typing import TypeVar, Callable

X = TypeVar("X")  # Generic type for data
Stat = TypeVar("Stat")  # Generic type for statistic


def bootstrap_sample(data: List[X]) -> List[X]:
    """randomly sample len(data) elements with replacement"""
    return [random.choice(data) for _ in data]


def bootstrap_statistic(data: List[X], stats_fn: Callable[[List[X]], Stat],
                        num_samples: int) -> List[Stat]:
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]


close_to_100 = [99.5 + random.random() for _ in range(101)]
far_from_100 = ([99.5 + random.random()] +
                [random.random() for _ in range(50)] +
                [200 + random.random() for _ in range(50)])

from scratch.statistics import median, standard_deviation

medians_close = bootstrap_statistic(close_to_100, median, 100)
medians_close
medians_far = bootstrap_statistic(far_from_100, median, 100)
medians_far


# Standard Errors of Regression Coefficients
# Taking bootstrap approach -> wide distribution == low confidence in beta
from typing import Tuple
import datetime

def estimate_sample_beta(pairs: List[Tuple[Vector, float]]):
    x_sample = [x for x, _ in pairs]
    y_sample = [y for _, y in pairs]
    beta = least_squares_fit(x_sample, y_sample, learning_rate, 5000, 25)
    print("bootstrap sample", beta)
    return beta

random.seed(0)
bootstrap_betas = bootstrap_statistic(list(zip(inputs, daily_minutes_good)),
                                      estimate_sample_beta, 100)

































