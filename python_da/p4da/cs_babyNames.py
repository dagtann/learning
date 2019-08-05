import numpy as np
import pandas as pd
import tqdm


## Load data
path = "/Users/dag/github/learning/python_da/p4da/babynames/"
colnames = ["name", "sex", "births"]
data_dictionary = {}
for year in tqdm.tqdm(range(1880, 2019)):
    data_dictionary[year] = pd.read_csv(path + f"yob{str(year)}.txt",
                                        index_col=None, header=None,
                                        names=colnames)
    data_dictionary[year]["year"] = year
babynames = pd.concat(data_dictionary, axis=0, ignore_index=True)

## Check property
babynames.info()
babynames.head()
babynames.describe()
list(map(lambda x: any(babynames[x].isnull()), colnames))
