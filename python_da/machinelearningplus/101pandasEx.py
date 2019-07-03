import numpy as np
import pandas as pd

# 101 Pandas Exercises

# 1 Show Pandas Version
pd.__version__
pd.show_versions(as_json=True)

# 2 Create a series from a list, numpy array, and dictionary
mylist = list("abcdefghijklmnopqrstuvwxyz")
myarr = np.arange(26)
mydict = dict(zip(mylist, myarr))
from_list = pd.Series(mylist)
from_array = pd.Series(myarr)
from_dictionary = pd.Series(mydict)

# 3 Convert series <ser> into a DataFrame, add its index as a column
data = pd.DataFrame(zip(from_dictionary, from_dictionary.index)) # my solution
data = from_dictionary.to_frame().reset_index()
data = from_dictionary.to_frame().reset_index(drop=True) # avoid saving old index

# 4 Combine multiple series into a dataframe
ser1 = pd.Series(list("abcdefghijklmnopqrstuvwxyz"))
ser2 = pd.Series(np.arange(len(ser1)))
data = pd.DataFrame({"ser1": ser1, "ser2": ser2}) # my solution
data = pd.concat([ser1, ser2], axis=0) # append ser1 by ser2
data = pd.concat([ser1, ser2], axis=1) # cbind ser1 by ser2


# 5 Assign a name to the series
ser = pd.Series(list("abcdefghijklmnopqrstuvwxyz"))
ser.name = "alphabets"
ser.head()

# 6 How to get the items of series A not present in series B?
ser1 = pd.Series(np.arange(1, 6))
ser2 = pd.Series(np.arange(4, 9))
ser1[~ser1.isin(ser2)]

# 7 How to get the items not common to both series A and B
ser1 = pd.Series(np.arange(1, 6))
ser2 = pd.Series(np.arange(4, 9))
ser1.index.difference(ser2)

serU = ser1.append(ser2)  # My solution
serU[~(serU.isin(ser1) & serU.isin(ser2))]

ser_u = pd.Series(np.union1d(ser1, ser2))  # Online solution
ser_i = pd.Series(np.intersect1d(ser1, ser2))
ser_u[~ser_u.isin(ser_i)]

# 9 Get the minimum, p25, p50, p75, and max of a numeric series
ser = pd.Series(np.random.normal(10, 5, 25))  # mu, sigma, N
ser.mean()
ser.var() ** (1 / 2)
len(ser)
fivepoint = [ser.quantile(p) for p in np.arange(0.0, 1.1, .25)]  # my solution
fivepoint = np.percentile(ser, q=np.arange(0, 101, 25))  # online solution

# 10 get the frequency counts of unique items of a series
ser = pd.Series(np.take(list("abcdefgh"), np.random.randint(8, size = 30)))
ser.value_counts()

# 10 How to keep only top 2 most frequent values as it is and replace everything
#   else as "Other"
np.random.RandomState(100)
ser = pd.Series(np.random.randint(1, 5, [12]))

top2 = pd.Series(ser.value_counts()[:2].index)  # my solution
ser[~ser.isin(top2)] = "Other"
ser

print("Top 2 Freq:", ser.value_counts())  # Online solution
ser[~ser.isin(ser.value_counts().index[:2])] = "Other"
ser

# 11 Bin a numeric series to 10 groups of equal size
ser = pd.Series(np.random.random(20))
ser_cut1 = pd.cut(ser, bins=10, retbins=False)  # my solution
pd.Series(ser_cut1).value_counts()  # does not ensure equally populated bins
ser_cut2 = pd.qcut(ser, q=np.arange(0, 1.1, .1),
                   labels=['1st', '2nd', '3rd', '4th', '5th', '6th', '7th',
                   '8th', '9th', '10th'])


# 12 Convert a numpy array to a dataframe of given shape
ser = pd.Series(np.random.randint(1, 10, 35))  # convert to shape=(7, 5)
pd.DataFrame(np.array(ser).reshape((7, 5)))  # my solution
pd.DataFrame(ser.values.res hape(7, 5))  # smarter solution
# 26 Get mean of a series grouped by another series
fruit = pd.Series(np.random.choice(["apple", "banana", "carrot"], 10))
weights = pd.Series(np.linspace(1, 10, 10))
print(weights.tolist())
print(fruit.tolist())
weights.groupby(fruit).mean()


# 13 Find the positions of numbers that are multiples of 3 from a series
ser = pd.Series(np.random.randint(1, 10, 7))
ser.index[(ser % 3) == 0]  # My solution: Boolean subsetting
np.argwhere(ser % 3 == 0)

import pandas as pd
import numpy as np

# 14
ser = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))
pos = [0, 4, 8, 14, 20]
ser2 = ser[pos]

# 15
ser1 = pd.Series(range(5))
ser2 = pd.Series(list('abcde'))
ser1.append(ser2)  # my solution
pd.concat([ser1, ser2], axis=0)
pd.concat([ser1, ser2], axis=1)

# 16
ser1 = pd.Series([10, 9, 6, 5, 3, 1, 12, 8, 13])
ser2 = pd.Series([1, 3, 10, 13])
ser1[ser1.isin(ser2)].index  # my solution

[np.where(i == ser1)[0].tolist()[0] for i in ser2]
[pd.Index(ser1).get_loc(i) for i in ser2]

# 17
truth = pd.Series(range(10))
pred = pd.Series(range(10)) + np.random.random(10)
pd.Series((pred - truth) ** 2).mean()  # my solution

np.mean((pred - truth) ** 2)

# 18
ser = pd.Series(["how", "to", "kick", "ass?"])
ser_capitalized = [s[0].upper() + s[1:] for s in ser]  # my solution
# draw back: returns a list, possibly informative index is lost

# better
ser.map(lambda x: x[0].upper() + x[1:])

# even better
ser.map(lambda x: x.title())

# 19
ser = pd.series(["how", "to", "kick", "ass"])
ser.map(lambda x: len(x))

# 20
ser = pd.Series([1, 3, 6, 10, 15, 21, 27, 35])
ser.diff(1).tolist()
ser.diff(2).tolist()
# 27 Compute the euclidean distance between two series
p = pd.Series(np.arange(1, 11))
q = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
sum((p - q) ** 2) ** (1 / 2)


# 28 find all the local maxima in a series
ser = pd.Series([2, 10, 3, 4, 9, 10, 2, 7, 3])
np.sign(np.diff(ser, 2))
dd = np.diff(np.sign(np.diff(ser)))
np.where(dd == -2)[0] + 1


# 29 Replace missing spaces with least frequence string
my_str = 'dbc deb abed gade'
freqs = {}
for s in my_str:
    freqs[s] = 1 + freqs.get(s, 0)

pd.Series(my_str).str.replace(" ", freqs.index[freqs == freqs.min()][0])
