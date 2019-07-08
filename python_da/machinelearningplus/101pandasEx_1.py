import numpy as np
import pandas as pd
import datetime

# 21
ser = pd.Series(["01 Jan 2010", "02-02-2011", "20120303", "2013/04/04",
                 "2014-05-05", "2015-06-06-T12:20"])
ser = pd.to_datetime(ser)

# 22 Get day of month, week number, day of year, day of week
print("Day of month:\n", list(ser.dt.day))
print("Week number:\n", list(ser.dt.weekofyear))
print("Day of year:\n", list(ser.dt.dayofyear))
print("Day of week:\n", list(ser.dt.weekday_name))

# 23 Convert year-month string to dates corresponding to 4th of months
ser = pd.Series(["Jan 2010", "Feb 2011", "Mar 2012"])
ser = pd.Series("04 " + ser)  # my solution
ser = pd.to_datetime(ser)

ser = pd.Series(["Jan 2010", "Feb 2011", "Mar 2012"])  # more elegant solution
ser.map(lambda x: pd.to_datetime("04 " + x))

ser = pd.Series(["Jan 2010", "Feb 2011", "Mar 2012"])
ser = pd.to_datetime(["04 " + x for x in ser])  # more pythonic solution

# 24 Filter words that contain at least 2 vowels
ser = pd.Series(["Apple", "Orange", "Plan", "Python", "Money"])


def count_vowels(s):  # my solution
    vowels = 0
    for c in s:
        if c.lower() in "aeiou":
            vowels += 1
    return vowels


ser[ser.map(count_vowels) >= 2]

# Filter valid emails from a list using regex
emails = pd.Series(['buying books at amazom.com', 'rameses@egypt.com',
                    'matt@t.co', 'narendra@modi.com'])
pattern ='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'
emails[emails.str.contains(pattern)]

# 26 Get mean of series grouped by another series
fruit = pd.Series(np.random.choice(["apple", "banana", "carrot"], 10))
weights = pd.Series(np.linspace(1, 10, 10))
weights.groupby(fruit).mean()

# 27 Compute the euclidian distances between two series
p = pd.Series(np.arange(1, 11))
q = p[::-1]
q.index = range(0,len(q))

sum((p - q) ** 2) ** (1 / 2)  # my solution
np.linalg.norm(p - q)  # using a function

# 28 Find all peaks in numeric series
ser = pd.Series([2, 10, 3, 4, 9, 10, 2, 7, 3])
# 1, 5, 7
ser[np.sign(ser.diff(2)) < 0].index - 1
ser.loc(np.diff(np.sign(np.diff(ser))) == -2)
np.diff(np.sign(np.diff(ser)))
