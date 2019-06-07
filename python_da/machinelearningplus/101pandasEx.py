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



