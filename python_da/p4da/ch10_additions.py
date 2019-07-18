import numpy as np
import pandas as pd
import string
import random


## 10.3 Apply: General split-apply-combine ------------------------------------
# GroupBy method apply(): most general-purpose method


def top(df, n=5, column=None):
    """Returns the top 5 entries from a DataFrame,
        sorted by column.
    """
    return df.sort_values(by=column)[-n:]

N = 250
df = pd.DataFrame({"data1": np.random.randn(N), "data2": np.random.randn(N),
                  "key1": random.choices(string.ascii_lowercase, k=N)})
top(df, column="data1")
df.groupby("key1").apply(top, n=5, column="data2")
# splits df by key1, finds top 5 rows, glues results together
# note how you pass non-default arguments
# single requirement for apply: returns pandas Series or DataFrame
# Inside group by, passing methods like describe is a short hand for:
# f = lambda x: x.describe()
# grouped.apply(f)

## Quantile and Bucket Analysis -----------------------------------------------
frame = pd.DataFrame({"data1": np.random.randn(1000),
                      "data2": np.random.randn(1000)})
# define bucket categorization
frame["quartiles"] = pd.qcut(frame["data1"], 4)
quartiles = pd.qcut(frame["data1"], 4)
frame["quartiles"][:10]
frame.groupby("quartiles")["data1"].mean()


def get_stats(group):
    return {"min": group.min(), "max": group.max(), "mean": group.mean(),
            "n": group.count()}
get_stats(group=frame["data2"])

grouped = frame.groupby("quartiles")
grouped["data2"].apply(get_stats)
grouped["data1"].apply(get_stats).unstack()

### Example: Fill NA with Group Specific Values
s = pd.Series(np.random.randn(6))
s[::2] = np.nan
s.fillna(s.mean())  # fill with series mean

# fill by group
states = ["Ohio", "New York", "Vermont", "Florida", "Oregon", "Nevada",
          "California", "Idaho"]
group_key = ["East"] * 4 + ["West"] * 4
data = pd.Series(np.random.randn(8), index = states)
data[["Vermont", "Nevada", "Idaho"]] = np.nan

fill_mean = lambda g: g.fillna(g.mean())
data.groupby(group_key).apply(fill_mean)

# fill NA by predefined fill value
fill_values = {"East": .5, "West": -1}
fill_func = lambda g: g.fillna(fill_values[g.name])
data.groupby(group_key).apply(fill_func)

### Example: Random Sampling and Permutation
# (1) Construct English-style playing cards
suits = ["H", "S", "C", "D"]
card_val = (list(range(1, 11)) + [10] * 3) * 4
base_names = ["A"] + list(range(2, 11)) + ["J", "K", "Q"]
cards = []
for suit in suits:
    cards.extend(str(num) + suit for num in base_names)
deck = pd.Series(card_val, index=cards)

# (2) draw a hand of 5 cards
random.choices(deck, k=5) #  does not work b/c index is dropped
def draw(deck, n=5):
    return deck.sample(n)
hand = draw(deck)
hand

# draw 2 random cards from each suit
get_suit = lambda card: card[-1]
deck.groupby(get_suit).apply(draw, n=2)
# ALTERNATIVE: drop hierarchical grouping index
deck.groupby(get_suit, group_keys=False).apply(draw, n=2)


### Example: Group Weighted Average and Correlation
df = pd.DataFrame({"category": ["a"] * 4 + ["b"] * 4,
                   "data": np.random.randn(8),
                   "weights": np.random.rand(8)})
df
df.groupby("category").apply(
    lambda g: np.average(g["data"], weights=g["weights"])
)

### Group Weighted Average and Correlation


### Example: Group-Wise Linear Regression
import statsmodels.api as sm
def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X["intercept"] = 1
    result = sm.OLS(Y, X).fit()
    return result.params

beta = pd.Series([.8, 1.24])
sigma = 1.5

N = 1000
X = pd.DataFrame({"intercept": [1] * N, "predvar": np.random.randn(N),
                  "group": ["a"] * 500 + ["b"] * 500})
Y = pd.Series(np.dot(X[["intercept", "predvar"]], beta) +
    np.random.normal(loc=0.0, scale=sigma, size=N), name = "depvar")
df = pd.concat([X[["predvar", "group"]], Y], axis = 1)
grouped = df.groupby("group")
grouped.apply(regress, "depvar", ["predvar"])

## 10.4 Pivot Tables and Cross-Tabulation
tips.

