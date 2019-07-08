import numpy as np
import pandas as pd

data = pd.DataFrame(np.random.rand(7, 5))
data.iloc[:4, 1] = np.nan
data.iloc[:2, 2] = np.nan

data.fillna({1: 0.5, 2: 1})
data.fillna({1: 0.5})  # NOTE: dict approach fills only stated cols

data = pd.Series([1., np.nan, 3.5, np.nan, 7.])  # fill mean of a series
data.fillna(data.mean())  # NOTE: Pandas removes NAs by default


# Data Transformation

# (a) Removing duplicates