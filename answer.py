"""100 numpy exercises.

This is a collection of exercises that have been collected in the numpy mailing
list, on stack overflow and in the numpy documentation. I've also created some
to reach the 100 limit. The goal of this collection is to offer a quick
reference for both old and new users but also to provide a set of exercises for
those who teach.

If you find an error or think you've a better way to solve some of them, feel
free to open an issue at <https://github.com/rougier/numpy-100>
"""

# -*- coding: utf-8 -*-

# %% 1. Import the numpy package under the name `np` (★☆☆)

import math
from io import StringIO

import numpy as np
import pandas as pd
import scipy.spatial
from numpy.lib import stride_tricks

# %% 2. Print the numpy version and the configuration (★☆☆)

np.__version__
np.show_config()

# %% 3. Create a null vector of size 10 (★☆☆)

np.zeros(10)

# %% 4.  How to find the memory size of any array (★☆☆)

Z = np.zeros((10, 10))
"{:d} bytes".format(Z.size * Z.itemsize)


# %% 5.  How to get the documentation of the numpy add function from the command line? (★☆☆)

"""
python -c "import numpy; numpy.info(numpy.add)"
"""

# %% 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆)

Z = np.zeros(10)
Z[4] = 1
Z

# %% 7.  Create a vector with values ranging from 10 to 49 (★☆☆)

np.arange(10, 50)

# %% 8.  Reverse a vector (first element becomes last) (★☆☆)

Z = np.arange(10, 50)
Z[::-1]

# %% 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)

np.arange(9).reshape((3, 3))

# %% 10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆)

np.nonzero([1, 2, 0, 0, 4, 0])

# %% 11. Create a 3x3 identity matrix (★☆☆)

np.eye(3)

# %% 12. Create a 3x3x3 array with random values (★☆☆)

np.random.random((3, 3, 3))

# %% 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)

Z = np.random.random((10, 10))
Z.min(), Z.max()

# %% 14. Create a random vector of size 30 and find the mean value (★☆☆)

Z = np.random.random(30)
Z.mean()

# %% 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)

Z = np.ones((10, 10))
Z[1:-1, 1:-1] = 0
Z

# %% 16. How to add a border (filled with 0's) around an existing array? (★☆☆)

# !!!
Z = np.ones((5, 5))
Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
Z


# %% 17. What is the result of the following expression? (★☆☆)

"""
```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
0.3 == 3 * 0.1
```
"""

# !!!
assert(math.isnan(0 * np.nan))
assert((np.nan == np.nan) is False)
assert((np.inf > np.nan)is False)
assert(math.isnan(np.nan - np.nan))
assert((0.3 == 3 * 0.1) is False)

# %% 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)

# !!!
Z = np.diag(1 + np.arange(4), k=-1)
Z

# %% 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)

Z = np.zeros((8, 8), dtype=int)
Z[::2, ::2] = 1
Z[1::2, 1::2] = 1
Z

# %% 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?

# !!!
np.unravel_index(100, (6, 7, 8))

# %% 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)

# !!!
Z = np.tile(np.eye(2), (4, 4))
Z

# %% 22. Normalize a 5x5 random matrix (★☆☆)

Z = np.random.random((5, 5))
Z_max, Z_min = Z.max(), Z.min()
Z = (Z - Z_min) / (Z_max - Z_min)
Z

# %% 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)

# !!!
color = np.dtype([('r', np.ubyte, 1),
                  ('g', np.ubyte, 1),
                  ('b', np.ubyte, 1),
                  ('a', np.ubyte, 1)])
color

# %% 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)

Z = np.ones((5, 3)) @ np.ones((3, 2))
Z

# %% 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)

Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1
Z

# %% 26. What is the output of the following script? (★☆☆)

"""
```python
# %% Author: Jake VanderPlas

print(sum(range(5), -1))
from numpy import *

print(sum(range(5), -1))
```
"""

"""
print(sum(range(5), -1))  # => 9
from numpy import *  # => overwrites `sum` func

print(sum(range(5), -1))  # => 10
"""

# %% 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)

"""
```python
Z**Z
2 << Z >> 2
Z < - Z
1j * Z
Z / 1 / 1
Z < Z > Z
```
"""

"""
Z**Z  # => illegal
Z < Z > Z  # => illegal
"""

# %% 28. What are the result of the following expressions?

"""
```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```
"""

assert(math.isnan(np.array(0) / np.array(0)))
assert((np.array(0) // np.array(0)) == 0)
np.array([np.nan]).astype(int).astype(float)  # => array([-9.22337204e+18])

# %% 29. How to round away from zero a float array ? (★☆☆)

a = np.random.uniform(-10, +10, 10)
Z = np.copysign(np.ceil(np.abs(a)), a)
Z

# %% 30. How to find common values between two arrays? (★☆☆)

a1 = np.random.randint(0, 10, 10)
a2 = np.random.randint(0, 10, 10)
Z = np.intersect1d(a1, a2)
Z

# %% 31. How to ignore all numpy warnings (not recommended)? (★☆☆)

# Suicide mode on# Suici
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0  # No warnings
# Back to sanity
_ = np.seterr(**defaults)
Z = np.ones(1) / 0  # Shows warning

# %% 32. Is the following expressions true? (★☆☆)

"""
```python
np.sqrt(-1) == np.emath.sqrt(-1)
```
"""

assert(math.isnan(np.sqrt(-1)))
assert(np.emath.sqrt(-1) == 1j)

# %% 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)

yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today = np.datetime64('today', 'D')
tomorrow = np.datetime64('today', 'D') + np.timedelta64(1, 'D')

# %% 34. How to get all the dates corresponding to the month of July 2016? (★★☆)

Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
Z

# %% 35. How to compute ((A+B)\*(-A/2)) in place (without copy)? (★★☆)

# ??? is this correct

A = np.ones(3) * 1
B = np.ones(3) * 2
np.add(A, B, out=B)
np.divide(A, 2, out=A)
np.negative(A, out=A)
np.multiply(A, B, out=A)
A

# %% 36. Extract the integer part of a random array using 5 different methods (★★☆)

Z = np.random.uniform(0, 10, 10)
print(Z - Z % 1)
print(np.floor(Z))
print(np.ceil(Z) - 1)
print(Z.astype(int))
print(np.trunc(Z))

# %% 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)

Z = np.zeros((5, 5))
Z += np.arange(5)  # broadcasting
Z

# %% 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)


def generate():
    for x in range(10):
        yield x


Z = np.fromiter(generate(), dtype=float, count=-1)
Z

# %% 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)

Z = np.linspace(0, 1, 11, endpoint=False)[1:]
Z

# %% 40. Create a random vector of size 10 and sort it (★★☆)

Z = np.random.random(10)
Z.sort()
Z

# %% 41. How to sum a small array faster than np.sum? (★★☆)

Z = np.arange(10)
np.add.reduce(Z)

# %% 42. Consider two random array A and B, check if they are equal (★★☆)

A = np.random.randint(0, 2, 5)
B = A
assert(np.allclose(A, B))
assert(np.array_equal(A, B))

# %% 43. Make an array immutable (read-only) (★★☆)

Z = np.zeros(10)
Z.flags.writeable = False
try:
    Z[0] = 1
except ValueError as e:
    print(e)

# %% 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)

Z = np.random.random((10, 2))
X, Y = Z[:, 0], Z[:, 1]
R = np.sqrt(X**2 + Y**2)
T = np.arctan2(Y, X)
np.array(list(zip(R, T)), float)

# %% 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)

Z = np.random.random(10)
print(Z)
Z[Z.argmax()] = 0
print(Z)

# %% 46. Create a structured array with `x` and `y` coordinates covering the \[0,1\]x\[0,1\] area (★★☆)

Z = np.zeros((5, 5), [('x', float), ('y', float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0, 1, 5),
                             np.linspace(0, 1, 5))
print(Z)

# %% 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))

X = np.arange(8)
Y = X + 0.5
# outer ... Apply the ufunc op to all pairs (a, b) with a in A and b in B.
C = 1.0 / np.subtract.outer(X, Y)
C

# %% 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)

for dtype in [np.int8, np.int32, np.int64]:
    print(np.iinfo(dtype))
for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype))

# %% 49. How to print all the values of an array? (★★☆)

po = np.get_printoptions()
po
np.set_printoptions(threshold=np.nan)
Z = np.zeros((32, 32))
print(Z)
np.set_printoptions(**po)
print(Z)

# %% 50. How to find the closest value (to a given scalar) in a vector? (★★☆)

Z = np.arange(100)
v = np.random.uniform(0, 100)
v
idx = (np.abs(Z - v)).argmin()
Z[idx]

# %% 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)

Z = np.zeros(10, [('position', [('x', float, 1),
                                ('y', float, 1)]),
                  ('color', [('r', int, 1),
                             ('g', int, 1),
                             ('b', int, 1)])])
Z

# %% 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)

Z = np.random.random((10, 2))
X, Y = np.atleast_2d(Z[:, 0], Z[:, 1])
D = np.sqrt((X - X.T)**2 + (Y - Y.T)**2)
D

# or

D = scipy.spatial.distance.cdist(Z, Z)
D

# %% 53. How to convert a float (32 bits) array into an integer (32 bits) in place?

Z = np.arange(10, dtype=np.float32)
Z = Z.astype(np.int32, copy=False)
Z

# %% 54. How to read the following file? (★★☆)

"""
```
1, 2, 3, 4, 5
6, , , 7, 8
    , , 9, 10, 11
```
"""

# Fake file
s = StringIO("""1, 2, 3, 4, 5\n
                6,  ,  , 7, 8\n
                 ,  , 9,10,11\n""")

np.genfromtxt(s, delimiter=',', dtype=np.int, filling_values=0)


# %% 55. What is the equivalent of enumerate for numpy arrays? (★★☆)

Z = np.arange(9).reshape(3, 3)
for idx, v in np.ndenumerate(Z):
    print(idx, v)
for idx in np.ndindex(Z.shape):
    print(idx, Z[idx])


# %% 56. Generate a generic 2D Gaussian-like array (★★☆)

X, Y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
D = np.sqrt(X * X + Y * Y)
sigma, mu = 1.0, 0.0
G = np.exp(-((D - mu)**2 / (2.0 * sigma**2)))
G


# %% 57. How to randomly place p elements in a 2D array? (★★☆)

n = 5
p = 3
Z = np.zeros((n, n))
np.put(Z, np.random.choice(range(n * n), p, replace=False), 1)
Z


# %% 58. Subtract the mean of each row of a matrix (★★☆)

Z = np.random.rand(5, 10)
Z = Z - np.mean(Z, axis=1, keepdims=True)
Z


# %% 59. How to sort an array by the nth column? (★★☆)

n = 3
Z = np.random.randint(0, 10, (5, 5))
Z
Z[Z[:, n].argsort()]


# %% 60. How to tell if a given 2D array has null columns? (★★☆)

Z = np.random.randint(0, 3, (3, 10))
Z
(~Z.any(axis=0)).any()


# %% 61. Find the nearest value from a given value in an array (★★☆)

Z = np.random.rand(10)
Z
x = 0.3
Z[np.abs(Z - x).argmin()]


# %% 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)

A = np.arange(3).reshape(3, 1)
B = np.arange(3).reshape(1, 3)
it = np.nditer([A, B, None])
for x, y, z in it:
    print(x, y)
    z[...] = x + y
# (0, 0), (0, 1), (0, 2), (1, 0), ... , (2, 2)
it.operands[2]


# %% 63. Create an array class that has a name attribute (★★☆)

class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, "name", "no name")


Z = NamedArray(np.arange(10), "range_10")
Z.name


# %% 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)

Z = np.ones(10)
I = np.random.randint(0, len(Z), 20)
Z += np.bincount(I, minlength=len(Z))
Z
# or
Z = np.ones(10)
I = np.random.randint(0, len(Z), 20)
np.add.at(Z, I, 1)
Z

# %% 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)

X = [1, 2, 3, 4, 5, 6]
I = [1, 3, 9, 3, 4, 1]
F = np.bincount(I, X)
F


# %% 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★)

w, h = 16, 16
I = np.random.randint(0, 2, (h, w, 3)).astype(np.ubyte)
F = I[..., 0] * 256 * 256 + I[..., 1] * 256 + I[..., 2]
Z = np.unique(F)
n = len(Z)
n


# %% 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)

A = np.random.randint(0, 10, (3, 4, 3, 4))
np.sum
s = A.sum(axis=(-2, -1))
s
# or
s = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
s


# %% 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★)

D = np.random.uniform(0, 1, 100)
S = np.random.randint(0, 10, 100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
D_means
# or
pd.Series(D).groupby(S).mean()


# %% 69. How to get the diagonal of a dot product? (★★★)

A = np.random.uniform(0, 1, (5, 5))
B = np.random.uniform(0, 1, (5, 5))

# slow
np.diag(A @ B)

# fast
np.einsum('ij,ji->i', A, B)


# %% 70. Consider the vector \[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)

Z = np.array([1, 2, 3, 4, 5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z) - 1) * (nz))
Z0[::(nz + 1)] = Z
Z0


# %% 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)

A = np.ones((5, 5, 3))
B = 2 * np.ones((5, 5))
Z = A * B[:, :, None]
Z


# %% 72. How to swap two rows of an array? (★★★)

A = np.arange(25).reshape(5, 5)
A[[0, 1]] = A[[1, 0]]
A


# %% 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★)

faces = np.random.randint(0, 100, (10, 3))
F = np.roll(faces.repeat(2, axis=1), -1, axis=1)
F = F.reshape(len(F) * 3, 2)
F = np.sort(F, axis=1)
G = F.view(dtype=[('p0', F.dtype), ('p1', F.dtype)])
G = np.unique(G)
G


# %% 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)

C = np.bincount([1, 1, 2, 3, 4, 4, 6])
Z = np.repeat(np.arange(len(C)), C)
Z


# %% 75. How to compute averages using a sliding window over an array? (★★★)

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)  # 累積和を取る
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

Z = moving_average(np.arange(20), 3)
Z


# %% 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\[0\],Z\[1\],Z\[2\]) and each subsequent row is  shifted by 1 (last row should be (Z\[-3\],Z\[-2\],Z\[-1\]) (★★★)

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
Z


# %% 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)

Z = np.random.randint(0, 2, 100)
np.logical_not(Z, out=Z)

# or

Z = np.random.uniform(-1.0, 1.0, 100)
np.negative(Z, out=Z)


# %% 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i  (P0\[i\],P1\[i\])? (★★★)

# n[1,...,1]            # equivalent to n[1,:,:,1]

def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:, 0] - p[..., 0]) * T[:, 0] +
          (P0[:, 1] - p[..., 1]) * T[:, 1]) / L
    U = U.reshape(len(U), 1)
    D = P0 + U * T - p
    return np.sqrt((D**2).sum(axis=1))

P0 = np.random.uniform(-10, 10, (10, 2))
P1 = np.random.uniform(-10, 10, (10, 2))
p = np.random.uniform(-10, 10, (1, 2))
Z = distance(P0, P1, p)
Z


# %% 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P\[j\]) to each line i (P0\[i\],P1\[i\])? (★★★)

P0 = np.random.uniform(-10, 10, (10, 2))
P1 = np.random.uniform(-10, 10, (10, 2))
p = np.random.uniform(-10, 10, (10, 2))
Z = np.array([distance(P0, P1, p_i) for p_i in p])
Z


# %% 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)

Z = np.random.randint(0, 10, (10, 10))
shape = (5, 5)
fill = 0
position = (1, 1)

R = np.ones(shape, dtype=Z.dtype) * fill
P = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop = np.array(list(shape)).astype(int)
Z_start = (P - Rs // 2)
Z_stop = (P + Rs // 2) + Rs % 2

R_start = (R_start - np.minimum(Z_start, 0)).tolist()
Z_start = (np.maximum(Z_start, 0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop - Zs, 0))).tolist()
Z_stop = (np.minimum(Z_stop, Zs)).tolist()

r = [slice(start, stop) for start, stop in zip(R_start, R_stop)]
z = [slice(start, stop) for start, stop in zip(Z_start, Z_stop)]
R[r] = Z[z]
print(Z)
print(R)


# %% 81. Consider an array Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], how to generate an array R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]? (★★★)


# %% 82. Compute a matrix rank (★★★)


# %% 83. How to find the most frequent value in an array?


# %% 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)


# %% 85. Create a 2D array subclass such that Z\[i,j\] == Z\[j,i\] (★★★)


# %% 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)


# %% 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)


# %% 88. How to implement the Game of Life using numpy arrays? (★★★)


# %% 89. How to get the n largest values of an array (★★★)


# %% 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)


# %% 91. How to create a record array from a regular array? (★★★)


# %% 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)


# %% 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)


# %% 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. \[2,2,3\]) (★★★)


# %% 95. Convert a vector of ints into a matrix binary representation (★★★)


# %% 96. Given a two dimensional array, how to extract unique rows? (★★★)


# %% 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)


# %% 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?


# %% 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)


# %% 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)
