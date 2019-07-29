import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def dnorm(mean, variance, size=1):
    if isinstance(size, int):
        size = size,
    return mean + variance ** (1 / 2) * np.random.randn(*size)

# For reproducibility
np.random.seed(12345)
N = 100
X = np.c_[dnorm(0, 0.4, size=N),
          dnorm(0, 0.6, size=N),
          dnorm(0, 0.2, size = N)]
eps = dnorm(0, 0.1, size=N)
beta = [0.1, 0.3, 0.5]

y = np.dot(X, beta) + eps

X_model = sm.add_constant(X)  # add intercept term to X
X_model[:5, ]

model = sm.OLS(y, X_model)  # Fit OLS regression
results = model.fit()
results.summary()
# access model components
results.params

# Alternative: Start from pandas and use patsy formula api
data = pd.DataFrame(X, columns = [f"x{i}" for i in range(X.shape[1])])
data["y"] = y
data.head()

results = smf.ols("y ~ x0 + x1 + x2", data=data).fit()
results.params
results.tvalues
results.summary()
# Note: (1) statsmodels returns pandas Series; (2) does not require manual
#   intercept; (3)
results.predict(data[:5])  # how2 make predictions

# time series analysis
# simulate time series data with autoregressive structure and noise
init_x = 4
import random
values = [init_x, init_x]
N = 1000

b0 = 0.8
b1 = -0.4
noise = dnorm(0, 0.1, N)
for i in range(N):
    new_x = values[-1] * b0 + values[-2] * b1 + noise[i]
    values.append(new_x)
# data has AR(2) structure with parameters 0.8 and -0.4

# iterative approximation of lag structure
maxlags = 5
model = sm.tsa.AR(values)
results = model.fit(maxlags)
results.params  # recovers AR structure

# Introduction to scikit-learn
train = pd.read_csv("")