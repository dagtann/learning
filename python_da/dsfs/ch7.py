# Hypothesis and Inference
from typing import Tuple
import math
from scratch.probability import normal_cdf

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """Returns mu and sigma corresponding to Binomial(n, p)"""
    mu = n * p
    sigma = math.sqrt(n * p * (1 - p))
    return mu, sigma


