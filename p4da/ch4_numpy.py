import numpy as np

# quick example: performance of np arrays compared to basic python data structures
my_arr = np.arange(1 * 10 ** 7)
my_list = list(range(1 * 10 ** 7))
%time for _ in range(10): my_arr2 = my_arr * 2
%time for _ in range(10): my_list2 = my_list * 2

# 4.1 NumPy ndarray
data = np.random.randn(2, 3)
data * 10
data + data
data * data

# ndarray = multidimensional, homogenous data
# shape: tuple indicating size of each dimension
data.shape
# dtype: attribute describing data type
data.dtype

# How to create ndarrays: use np.array() function, accepts any sequence-like object
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)

# special array creation functions
np.zeros(10) # 1dim array containing 10 0s
np.zeros((3, 6)) # 2dim array cotaining 0 in 3 rows and 6 columns
np.zeros((3, 6, 9)) # 3dim array cotaining 0s in 6 rows, 9 columns, and 3 slices
np.arange(10)
np.ones_like(data) # ndarray of 1s with shape data.shape
np.eye(3)