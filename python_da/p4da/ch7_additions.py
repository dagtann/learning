import pandas as pd
import numpy as np

## Replacing values
# use method method replace
data = pd.Series([1, -999, 2, -999, -1000, 3.])
data.replace(-999, np.nan) # single replacement
data.replace([-999, -1000], np.nan)  # replace multiple values
data.replace([-999, -1000], [np.nan, 0])  # multiple replacement values
data.replace({-999: np.nan, -1000:0})  # use dict to replace

## Renaming axis indexes
data = pd.DataFrame(np.arange(12).reshape((3,4)),
                    index=["Ohio", "Colorado", "New York"],
                    columns=["one", "two", "three", "four"])
transform = lambda x: x[:4].upper()  # cut to 4 chars and convert to upper
data.index.map(transform)  # transform is mapped to each index element

# NOTE: You can assign to index and hence modify inplace
data.index = data.index.map(transform)
data.index

# Use method rename to avoid modifying the original data
data.rename(index=str.title, columns=str.upper)
# Use rename with a dict like object to provide new labels for subset of indexes
data.rename(index={"OHIO": "INDIANA"}, columns={"three": "peekaboo"})
# add argument inplace=True to modify in place

# Discretization and binning
# function cut(<data>, <bins>, right=True):
# discretize <data> into categories <bins>, right inclusive
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)
cats
cats.value_counts()
# cut returns a Categorical object which can be treated like an array of strings
cats.codes
cats.categories

group_names = ["Youth", "YoungAdult", "MiddleAged", "Senior"]
pd.cut(ages, bins, labels=group_names)

data = np.random.rand(20)
pd.cut(data, 4, precision = 2)
# use integer i to return i categories of roughly equal width
# precision limits the number of significant digits to take into account

data = pd.Series(np.random.randn(1000))  # 1000 draws from N(0, 1)
data.describe()
cats = pd.qcut(data, 4)  # cut into quartiles
cats.value_counts()  # NOTE: bins are equally populated


## Detect and Filter outliers
data = pd.DataFrame(np.random.randn(1000, 4))
data.describe()

# detect outliers in a single series
col = data[2]
col[np.abs(col) > 3]

# detect any row in data that has outliers
data[(np.abs(data) > 3).any(1)]
# cap outliers
data[(np.abs(data) > 3).any(1)] = np.sign(data) * 3
data.describe()

# Permutation and Random Sampling
df = pd.DataFrame(np.arange(5 * 4).reshape((5, 4)))
df.describe()

sampler = np.random.permutation(5)
# returns an array of randomly permuted integers 0 to 5
df.iloc[sampler]
df.take(sampler)

# take a random sample WITHOUT replacement
df.sample(n=3)
# take a random sample WITH replacement
choices = pd.Series([5, 7, -1, 6, 4])
draws = choices.sample(n=10, replace=True)
draws  # Note the duplicated indexes


# Compute Indicator Variables
df = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "b"], "data1": range(6)})
pd.get_dummies(df["key"])
pd.get_dummies(df["data1"])

pd.get_dummies(df["key"], prefix="key")  # add a prefix
df_with_dummy = df[["data1"]].join(pd.get_dummies(df["key"]))
# NOTE: double bracket selection return a DataFrame
#   single bracket selection returns a Series
df_with_dummy


# combine indicator variables with discretization
np.random.seed(123456)
values = pd.Series(np.random.randn(1000))
values.describe()
df = pd.DataFrame(values, columns = ["value"])

quartile_names = ["q1", "q2", "q3", "q4"]
values_quartiles = pd.qcut(values, 4, quartile_names)
df = df.join(pd.get_dummies(values_quartiles))
df.head()


# String Manipulation
val = "a,b,  guido"
val.split(",")  # split string by <char>
[x.strip() for x in val.split(",")]  # trim trailing white space
pieces = [x.strip() for x in val.split(",")]
first, second, third = pieces  # unpack string elements
first + "::" + second + "::" + third  # join strings by "::"
"::".join(pieces)  # more generic method

"guido" in val  # find substring
val.index(",")  # return index of first match, raises exception if <s> not found
val.index("guido") == val.index("g")  # Note: returns index of 1st match in <s>
val.find(":")  # returns -1 if <s> not found
val.rfind(",") # returns index of last occurrence of <s> and -1 if not found
val.count(",")  # return no. of occurences of <s>
val.replace(",", "::")  # replace <s> by <s'>
val.replace(",", "")  # leave <s'> empty to delete pattern

## regular expressions
# located in python's re module
import re
text = "foo    bar\t   \tqux"
re.split("\s+", text) # goal: split at white space

regex = re.compile("\s+")  # compile a reusable re object, saves cpu cycles on
# repeated application
regex.split(text)
regex.findall(text)  # return a list of ALL matches in <string>
regex.search(text)  # return the first match
regex.match(text)  # match only at the beginning of a string

# example using email addresses
text = """
Dave dave@googlemail.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@gmail.com
"""

re_pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
regex = re.compile(re_pattern, flags = re.IGNORECASE)  # compile case-insensitive pattern
regex.findall(text)
m = regex.search(text)  # returns object w/i beginning and ending of first match
text[m.start():m.end()]
regex.match(text) == None  # returns None b/c matches only at the start of <s>
print(regex.sub("REDACTED", text))  # substitute <pattern> for matches in <s>

re_pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex = re.compile(re_pattern)
m = regex.match("wesm@bright.net")
m.groups()