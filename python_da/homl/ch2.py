import os
import tarfile
from six.moves import urllib
import pandas as pd

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(HOUSING_PATH):
        os.makedirs(HOUSING_PATH)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.info()
housing.describe(include="all")
housing["ocean_proximity"].unique()
housing["ocean_proximity"].value_counts()

%matplotlib
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))

# create a test set
import numpy as np


def split_train_data(data, test_ratio):
    idx_urn = np.random.uniform(size=data.shape[0])
    train = data[idx_urn > test_ratio]
    test = data[idx_urn <= test_ratio]
    return train, test

train_set, test_set = split_train_data(housing, .2)
assert len(test_set) + len(train_set) == len(housing)
# problem:
    # 1. new split over on each execution.
    #   program eventually sees the entire training set.
    # 2. Using np.random.seed() breaks as soon as the data updates
    # -> Use hashs of each instances identifier

# split data using sklearn
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# stratified sampling

housing.median_income.hist()

# (1) discretize median_income
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["median_income"] < 5, .0, inplace=True)

# (2) StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
housing["income_cat"].value_counts() / len(housing)
strat_train_set["income_cat"].value_counts() / len(strat_train_set)
strat_test_set["income_cat"].value_counts() / len(strat_test_set)

# (3) remove categorical attribute
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)


