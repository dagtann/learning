import json
import numpy as np
import pandas as pd

from collections import Counter
import seaborn as sns

path = "./pydata-book-2nd-edition/datasets/bitly_usagov/example.txt"
open(path).readline()
records = [json.loads(line) for line in open(path)]
type(records)
records[:5]

# distribution of time zones in data using python base library
time_zones = [record["tz"] for record in records if "tz" in record.keys()]
time_zones[:10]
tz_counts = {tz: time_zones.count(tz) for tz in time_zones}
len(time_zones)  # there are 3440 entries w/i tz in the data
len(tz_counts)  # users came from at least 97 different time zones


def top_counts(count_dict, n):  # show n most frequent time zones
    value_key_pairs = [(value, key) for key, value in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]


top_counts(tz_counts, 10)
Counter(time_zones).most_common(10)  # simplified version using Counter


# distribution of time zones in data using pandas
tz_frame = pd.DataFrame(records)
tz_frame.info()

tz_counts = tz_frame["tz"].value_counts()

tz_frame["tz"].describe()
sum(tz_frame["tz"].isnull())  # 120 missing entries in tz

clean_tz = tz_frame["tz"].fillna("Missing")
clean_tz[clean_tz == ""] = "Unknown"
tz_counts = clean_tz.value_counts()
tz_counts

subset = tz_counts[:10]
sns.barplot(y = subset.index, x = subset.values)

tz_counts[:10]
tz_frame["tz"].value_counts()

# parsing agent information
frame = pd.DataFrame(records)
frame["a"][50]
frame["a"][51]
results = pd.Series([x.split()[0] for x in frame["a"].dropna()]) # extract browser
results.value_counts()
cframe = frame[frame.a.notnull()]
os = pd.Series(["Windows" if "Windows" in c else "not Windows"
                for c in cframe["a"]])  # base pandas
os.name = "os"
cframe = pd.concat([cframe, os], axis = 1)
cframe.head()
# cframe["os"] = np.where(cframe["a"].str.contains("Windows"),  # np alternative
#                         "Windows", "Not Windows")

by_tz_os = cframe.groupby(["tz", "os"])  # group data
agg_counts = by_tz_os.size().unstack().fillna(0)
agg_counts
# indexer = agg_counts.sum(1).argsort()
# indexer
# count_subset = agg_counts.take(indexer[-10:])
count_subset = agg_counts.sum(1).nlargest(10)  # pandas alternative 10 largest
count_subset.name = "total"
agg_counts = agg_counts.stack()
agg_counts = pd.concat([agg_counts, count_subset], axis=1, sort=True)
agg_counts[:10]