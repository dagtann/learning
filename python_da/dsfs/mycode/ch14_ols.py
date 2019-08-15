# Assume population model y_i = beta * x_i + alpha + error

def predict(alpha: float, beta: float, x_i: float) -> float:
    return beta * x_i + alpha


def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    return y_i - predict(alpha, beta, x_i)

from scratch.linear_algebra import Vector
def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))


# Find least squares fit
from typing import Tuple
from scratch.statistics import correlation, standard_deviation, mean


def least_squares_fit(x: Vector, y: Vector) -> Tuple[float, float]:
    """
    Given vectors x and y,
    find the least-squares values of alpha and theta
    """
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta


# Quick test
a = -5; b = 3
x = [i for i in range(-100, 110, 10)]
y = [a + b * x_i for x_i in x]
assert least_squares_fit(x, y) == (a, b) # works

from scratch.statistics import num_friends_good, daily_minutes_good
alpha, beta = least_squares_fit(num_friends_good, daily_minutes_good)
assert 22.9 < alpha < 23
assert 0.9 < beta < .905

# plot predictions
yhats = [predict(alpha, beta, x_i) for x_i in num_friends_good]

import matplotlib.pyplot as plt
plt.plot(num_friends_good, yhats)
plt.scatter(num_friends_good, daily_minutes_good)
plt.xlabel("# of friends")
plt.ylabel("minutes per day")
plt.title("Simple linear regression model")
plt.show()

# Goodness of fit
from scratch.statistics import de_mean


def total_sum_of_squares(y: Vector) -> float:
    return sum(y_i ** 2 for y_i in de_mean(y))
total_sum_of_squares(y)

def r2(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return 1 - sum_of_sqerrors(alpha, beta, x, y) / total_sum_of_squares(y)

r2(alpha, beta, num_friends_good, daily_minutes_good)


# Using Gradient descent instead of normal equations
import random
import tqdm
from scratch.gradient_descent import gradient_step

num_epochs = 10000
random.seed(0)

guess = [random.random(), random.random()]  # initialize alpha and beta

learning_rate = 0.00001

with tqdm.trange(num_epochs) as t:
    for _ in t:
        alpha, beta = guess

        # Partial derivative of loss w.r.t. alpha
        grad_a = sum(2 * error(alpha, beta, x_i, y_i)
                     for x_i, y_i
                     in zip(num_friends_good, daily_minutes_good))
        # Partial derivative w.r.t. beta
        grad_b = sum(2 * error(alpha, beta, x_i, y_i) * x_i
                     for x_i, y_i
                     in zip(num_friends_good, daily_minutes_good))
        loss = sum_of_sqerrors(alpha, beta, num_friends_good, daily_minutes_good)
        t.set_description(f"loss: {loss:.3f}")
        # update guess
        guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)
