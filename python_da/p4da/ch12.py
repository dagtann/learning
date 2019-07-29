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
df["fruit"] = df["fruit"].astype("category")  # type conversion by assignment
my_categores = pd.Categorical(["foo", "bar", "baz", "foo", "bar"])  # categorical by declaration
my_categores.codes

# example: from_codes constructor
categories = ["foo", "bar", "baz"]
codes = [0, 1, 2, 0, 0, 1]
my_categories_2 = pd.Categorical.from_codes(codes, categories)
my_categories_2
# use ordered=True to indicate ordered categories
# as method as_ordered() to convert unordered to ordered cateforical data
# categorical arrays can consist of any immutable value types
pd.Categorical([1, 2, 1, 1])

# Computation with categoricals
np.random.seed(12345)
draws = np.random.randn(1000)
draws[:5]
bins = pd.qcut(draws, 4)  # compute quartile binning
bins
bins = pd.qcut(draws, 4, labels=[f"Q{i}" for i in range(1, 5)])  # assign different labels
bins


"".join(["1", "2", "3", "4"])