# Ch 6: Probability
import enum, random
from matplotlib import pyplot as plt
from Collections import Counter

## Conditional Probability
class Kid(enum.Enum):
    BOY = 0
    GIRL = 1


def random_kid() -> Kid:
    """ Draw a kid of random gender """
    return random.choice([Kid.BOY, Kid.GIRL])



both_girls = 0
older_girl = 0
either_girl = 0
N = 10000

random.seed(0)

for _ in range(N):
    younger = random_kid()
    older = random_kid()
    if older == Kid.GIRL:
        older_girl += 1
    if older == Kid.GIRL and younger == Kid.GIRL:
        both_girls += 1
    if older == Kid.GIRL or younger == Kid.GIRL:
        either_girl += 1

print("P(both | older):", both_girls / older_girl)
print("P(both | either):", both_girls / either_girl)

## Bayes' Theorem
# P(A | B) = P(A, B) / P(B) -> P(A, B) = P(A | B) P(B)
# P(B | A) = P(A, B) / P(A) -> P(A, B) = P(B | A) P(A)
# P(A | B) P(B) = P(B | A) P(A)
# P(A | B) = P(B | A) P(A) / P(B) where P(B) = P(B | A) + P(B | ~A)

p_T_D = 0.99
p_D = 1 / 10000
p_T_d = 1 - p_T_D
p_d = 1 - P_D

p_T_D * p_D / (p_T_D * p_D + p_T_d * p_d)

## Random Variables
### Continuous distributions

#### Uniform distribution
def uniform_pdf(x: float, min: float = 0, max: float = 1) -> float:
    """ probability density function of the uniform distribution """
    return 1 if min <= x < max else 0


def uniform_cdf(x: float, min: float = 0, max: float = 1) -> float:
    if x < min:
        return 0
    elif x < max:
        return x
    return 1


xs = [_ / 10 for _ in range(-10, 21)]
ys = [uniform_cdf(x) for x in xs]

plt.plot(xs, ys, linestyle = "solid")
plt.title("The uniform cdf")
plt.show()


### Normal distribution
import math
SQRT_TWO_PI = math.sqrt(2 * math.pi)
SQRT_TWO_PI


def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (math.exp(-1 * (x - mu) ** 2 / (2 * sigma ** 2)) /
            (SQRT_TWO_PI * sigma))


xs = [_ / 10 for _ in range(-50, 51)]

plt.plot(xs, [normal_pdf(x) for x in xs], "-", label="mu=0, sigma=1")
plt.plot(xs, [normal_pdf(x, 0, 3) for x in xs], "--", label="mu=0, sigma=3")
plt.plot(xs, [normal_pdf(x, 0, 1 / 3) for x in xs], ":", label="mu=0, sigma=1/3")
plt.legend()
plt.title("Various Normal PDFs")
plt.show()


def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return(1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2


plt.plot(xs, [normal_cdf(x) for x in xs], "--", label="mu=0, sigma=1")
plt.plot(xs, [normal_cdf(x, 0, 3) for x in xs], "--", label="mu=0, sigma=3")
plt.plot(xs, [normal_cdf(x, 0, 1 / 3) for x in xs], "--", label="mu=0, sigma=1/3")
plt.title("Various Normal CDFs")
plt.legend()
plt.show()


def inverse_normal_cdf(p: float,
                       mu: float = 0,
                       sigma: float = 1,
                       tol: float = 1e-5) -> float:
    """ Find approximate inverse using binary search """
    # standardize if not standard
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tol=tol)
    lo_z = -10.0
    hi_z = 10.0
    while hi_z - lo_z > tol:
        mid_z = (lo_z + hi_z) / 2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            lo_z = mid_z
        else:
            hi_z = mid_z
    return mid_z


inverse_normal_cdf(.975, tol=1e-6)

### Central limit theorem
# If x_1, ..., x_n are random variables with mean mu and stadnard deviation
# sigma, then 1/n sum_n(x) is approximately normal with standard deviation
# sigma / sqrt(n). Equivalently: (sum_n(x) - mu * n) / (sigma * sqrt(n)) is
# approximately standard normal.

#### Illustration using Bernoulli trials


def bernoulli_trial(p: float) -> int:
    "Returns 1 with probability p and 0 otherwise"
    return 1 if random.random() < p else 0


N = int(10e3)
p = 0.5
trials = [bernoulli_trial(p) for _ in range(N)]
sum(trials) / N


def binomial(n: int, p: int) -> int:
    return sum([bernoulli_trial(p) for _ in range(n)])


# plot a binomial histogram to demonstrate convergence on the normal


def binomial_histogram(p: float, n: int, num_points: int) -> float:
    """Pick points from a Binomial(n, p) and plot their distribution"""
    data = [binomial(n, p) for _ in range(num_points)]

    # use a bar chart to show the actual binomial samples
    histogram = Counter(data)
    plt.bar([x - .4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8,
            color="0.75")

    # parameters of the normal distribution
    mu = p * n
    sigma = math.sqrt((1 - p) * p * n)

    # use a line chart to show the normal distribution
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(x + 0.5, mu, sigma) -
          normal_cdf(x - 0.5, mu, sigma) for x in xs]
    plt.plot(xs, ys)
    plt.title("Binomial Distribution vs. Normal Approximation")

binomial_histogram(.75, 100, 10e3)
