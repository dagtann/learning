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

# Computations with Categoricals
np.random.seed(12345)
draws = np.random.randn(1000)
draws[:5]
# compute quartile binning
bins1 = pd.qcut(draws, 4)
bins1
bins2 = pd.qcut(draws, 4, labels = [f"Q{i}" for i in range(1, 5)])
bins2
bins2 = pd.Series(bins2, name = "quartile")
results = (pd.Series(draws)
           .groupby(bins2)  # use quartiles as groups
           .agg(["count", "min", "max"])  # aggregate information
           .reset_index())
results

# Better performance with categoricals
N = int(10e6)
draws = pd.Series(np.random.randn(N))
labels = pd.Series(["foo", "bar", "baz", "qux"] * (N // 4))
categories = labels.astype("category")
labels.memory_usage()
categories.memory_usage()
# GroupBy operations on categorical data may be faster because underlying
# algorithms integer-based code arrays instead of strings

# Categorical Methods
s = pd.Series(["a", "b", "c", "d"] * 2)
cat_s = s.astype("category")
cat_s.cat.codes  # attribute cat provides access to categorical methods
cat_s.cat.categories
# use set_categories() method to modify categories
actual_categories = ["a", "b", "c", "d", "e"]
cat_s2 = cat_s.cat.set_categories(actual_categories)
cat_s2
cat_s2.value_counts()

# use remove_unused_categories to trim unobserved categories
cat_s3 = cat_s[cat_s.isin(["a", "b"])]
cat_s3.cat.remove_unused_categories()

# create dummy variables for modeling
cat_s = pd.Series(["a", "b", "c", "d"] * 2, dtype="category")
pd.get_dummies(cat_s)  # use function get_dummies to create dummy indicators
