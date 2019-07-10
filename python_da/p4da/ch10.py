import numpy as np
import pandas as pd

# Data Aggregation and Group Operations =======================================
df = pd.DataFrame({"key1": ["a", "a", "b", "b", "a"],
                   "key2": ["one", "two", "one", "two", "one"],
                   "data1": np.random.randn(5),
                   "data2": np.random.randn(5)})

## 10.1 GroupBy Mechanics =====================================================

### group & aggregate a pd.Series ---------------------------------------------
grouped = df["data1"].groupby(df["key1"])  # returns pandas.core.groupby.SeriesGroupBy object
grouped.mean()  # result has index name key1 b/c var grouped by

# multiple key aggregation
means = df["data1"].groupby([df["key1"], df["key2"]]).mean()
# multiple keys of length len(data1) are passed as list
# resulting grouped series has hierachical index
means
means = df["data1"].groupby([df["key2"], df["key1"]]).mean()
means  # order of hierarchy changes with order of key assignment
means.unstack()

# keys can be any array of right length
states = np.array(["Ohio", "California", "California", "Ohio", "Ohio"])
years = np.array([2005, 2005, 2006, 2005, 2006])
df["data1"].groupby([states, years]).mean()
pd.concat([pd.Series(states), pd.Series(years)], axis = 1).shape()

# if grouping information is stored in the same dataset:
df.groupby("key1").mean()  # drops key2 b/c column is not numeric
df.groupby(["key1", "key2"]).mean()

# query size of each group:
df.groupby(["key1", "key2"]).size()
# Any missing values from group key will be dropped

### Iterating over groups -----------------------------------------------------
# Syntactic sugar:
df["data1"].groupby(df["key1"])
# is simplified by
df.groupby("key1")["data1"]

### Subset aggregation: -------------------------------------------------------
df.groupby(["key1", "key2"])[["data2"]].mean()  # returns grouped DataFrame

### Grouping with Dicts and Series --------------------------------------------
people = pd.DataFrame(np.random.randn(5, 5),
                      columns=["a", "b", "c", "d", "e"],
                      index=["Joe", "Steve", "Wes", "Jim", "Travis"])
mapping = {"a": "red", "b": "red", "c": "blue", "d": "blue", "e": "red",
           "f": "orange"}
# Mapping defines a correspondence between columns in people and colour groups.
# Note: key "f" is not used in people. groupy will work nonetheless.
by_column = people.groupby(mapping, axis=1)
by_column.sum()

map_series = pd.Series(mapping) # use a Series to group columns
map_series
people.groupby(map_series, axis=1).count()

### Grouping with Functions ---------------------------------------------------
# Python functions can be used as  group keys. The function will be called once
# per index value, with the return values being used as group names.
people.groupby(len, axis = 0).sum()
people.groupby(len, axis = 1).sum()

# functions can be mixed with arrays, dicts, or Series
key_list = ["one", "one", "one", "two", "two"]
people.groupby([len, key_list]).min()

### Grouping by Index levels --------------------------------------------------
columns = pd.MultiIndex.from_arrays([["US", "US", "US", "JP", "JP"],
                                    [1, 3, 5, 1, 3]],
                                    names=["cty", "tenor"])
hier_df = pd.DataFrame(np.random.randn(4, 5), columns=columns)
hier_df
# grouping by level: pass keyword <level>
hier_df.groupby(level="cty", axis=1).count()

## 10.2 Data Aggregation ======================================================