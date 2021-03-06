import numpy as np

# quick ex: performance of np arrays compared to basic python data structures
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

# How to create ndarrays: use np.array() function
# accepts any sequence-like object
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)

# special array creation functions
np.zeros(10)  # 1dim array containing 10 0s
np.zeros((3, 6))  # 2dim array cotaining 0 in 3 rows and 6 columns
np.zeros((3, 6, 9))  # 3dim array cotaining 0s in 6 rows, 9 columns, and 3 slices
np.arange(10)
np.ones_like(data)  # ndarray of 1s with shape data.shape
np.eye(3)

# Arithmetic with NumPy Arrays
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
arr * arr  # element by element multiplication
arr - arr  # element by element substraction
1 / arr  # scalar multiplication
arr ** 0.5  # square root of each element

# Boolean comparisons
arr2 = np.array([[0, 4, 1], [7, 2, 12]], dtype=float)
arr == arr2 # returns boolean array of equal size if arr1.shape == arr2.shape
arr2 > arr

# Basic indexing & slicing
arr = np.arange(10)
arr[5]  # select element no. 5
arr[5:8]  # select 5 up to 8
arr[5:8] = 12  # replace 5 through 7 inclusive with 12
# scalar 12 is broadcaseted (recycled)
# ATTENTION: SLICES ARE VIEWS, NOT NEW ASSIGNMENTS
# -> MODIFICATIONS WILL CHANGE THE ORIGINAL SOURCE ARRAY
arr_slice = arr[5:8]
arr_slice
arr_slice[1] = 12345
arr
# To explicitly copy a slice use method copy()
arr_slice = arr[5:8].copy()
arr_slice[:] = 999
arr
arr_slice

# select and slice ndim arrays
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2][2]  # recursive indexing
# to avoid recursive indexing use comma separated lists of indices
arr2d[2, 2]  # [axis 0, axis 1] -> [rows, columns]
# if later indices in multidimensional arrays are omitted, the returned object
# will be a lower dimensional ndarray consinsting of all the data along higher
# dimensions
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d
arr3d[0]
arr3d[1]

old_values = arr3d[0].copy()
arr3d[0] = 43
arr3d[0] = old_values

arr3d[1, 0]  # return all elements whose index starts with 1

# Indexing with slices
arr[1:6]

arr2d[:2]  # slices along axis 0
arr2d[:2, 1:]  # extract rows 0 to 1 and columns 1 through J
arr2d[1, :2]  # extract row 1, columns 0 up to 2
arr2d[:, :1]  # extract all rows and column 0
arr2d[:2, 1:] = 0  # replace elements in rows 0 and 1 in columns 1:J by 0
arr2d


# Boolean Indexing
names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joe"])
data = np.random.randn(7, 4)
names
data
len(names)
names == "Bob"
data[names == "Bob"] # indexes rows 0 and 4
# boolean array must be of the same length as the array axis it’s indexing,
# but indexing will NOT fail if the boolean array is not the correct length

data[names == "Bob", 2:]
data[names != "Bob"]  # select everything but Bob
data[~(names != "Bob")]  # select nothing but Bob, ~ negates the condition
# combine conditions using &, |
mask = (names == "Bob") | (names == "Will")
data[mask]
# set values with boolean arrays
data[data < 0] = 0
data[names != "Joe"] = 7

# Fancy indexing
# = indexing using integer arrays
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i  # replace axis 0 position i
# select a subset of rows
arr[[4, 3, 0, 6]]
arr[[-3, -5, -7]]  # select rows from the end

# ATTENTION: When given multiple index arrays, NumPy will create tuples
# The result of such indexing will ALWAYS be one-dimensional
arr = np.arange(32).reshape((8, 4))
arr[[1, 5, 7, 2], [0, 3, 1, 2]]  # equiv to (1, 0), (5, 3), ...

# transposing arrays and swapping axes
# transposition = special case of reshape
arr = np.arange(15).reshape((3, 5))
arr
arr.T  # Transpose

arr = np.random.randn(6, 3)
np.dot(arr, arr) # won't work
np.dot(arr, arr.T) # will work

# for higher dimensional arrays transpose accepts a TUPLE of axis numbers to
# permute axes
arr = np.arange(16).reshape((2, 2, 4))
arr.transpose(1, 0, 2) # axis reordered as stated, method .T is special case (1, 0)

arr = np.arange(16).reshape((8, 2))
(arr.T == arr.transpose(1, 0)).all()

# Universal functions ufunc
#   is: function that performs vectorized operations on ndarrays
#   examples: sqrt, exp
arr = np.arange(10)
np.sqrt(arr)
np.exp(arr)

# unary functions allow an optional out argument, allowing in-place operations
arr = np.random.randn(7) * 5
np.sqrt(arr, arr)

# Array-oriented programming with arrays

# Example evaluate (x ^ 2 + y ^ 2) ^ (1 / 2) across a regular grid of values
points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
z = np.sqrt(np.square(xs) + np.square(ys))

points = np.arange(4)
xs, ys = np.meshgrid(points, points)
z = np.sqrt(np.square(xs) + np.square(ys))

import matplotlib.pyplot as plt
plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
plt.show()

# Random number operations
# Example: Random walk. Starting from 0 add 1 with probability .5 and -1 otherwise
import random
np.seed(1999)
position = 0.0
n_steps = int(1e3)
walk = list()
for i in range(n_steps):
    cond = random.randint(0, 1)
    print("Condition:", cond)
    step = 1 if cond else -1
    print("Step:", step)
    position += step
    walk.append(position)
plt.plot(walk)
plt.show()