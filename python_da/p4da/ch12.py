import pandas as pd
import numpy as np

# 12.1 Categorical Data
s = pd.Series([1, 2, 2, 3, 3, 3, 5])
s.unique()
pd.unique(s)
pd.value_counts(s)

# dimension table
values = pd.Series([0, 1, 0, 0] * 2)
dim = pd.Series(["apple", "orange"])
dim.take(values) # uses take to restore an original series of strings
# take(self, indices[, axis, is_copy])
#       Return the elements in the given positional indices along an axis.

fruits = ["apple", "orange", "apple", "apple"] * 2
N = len(fruits)
df = pd.DataFrame({"fruit": fruits,
                   "basket_id": np.arange(N),
                   "count": np.random.randint(3, 15, size=N),
                   "weight": np.random.uniform(0, 4, size=N)},
                   columns=["basket_id", "fruit", "count", "weight"])
fruit_cat = df["fruit"].astype("category").values
fruit_cat
fruit_cat.categories
fruit_cat.codes
df["fruit"] = df["fruit"].astype("category")