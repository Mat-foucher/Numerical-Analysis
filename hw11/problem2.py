
import numpy as np
from numpy.random import default_rng
import sys
import math
import matplotlib.pyplot as plt
import scipy.integrate 
from scipy.interpolate import lagrange
import sympy as sym

from sympy import init_printing
from scipy.interpolate import BarycentricInterpolator
from scipy.integrate import quad
from scipy.linalg import hilbert
import time
from collections import OrderedDict
from scipy.special import roots_laguerre

import pandas as pd
init_printing()

import scipy.special.lambertw as W







# Gaussian Elimination and Back Substitution sans pivoting:

# n := number of unknowns, A := augmented system, 
def gaussElim(n, A):
  # Test if the array is square:
  assert len(A) == len(A[0]) - 1

  # Solution Vector:
  s = np.zeros(len(A))

  # Gaussian Elimination:
  for i in range(len(A)):
    # Optimistically Assume a zero pivot is never encountered.
    for j in range(i+1 , n):
      r = (A[j][i])/(A[i][i])
      for k in range(n+1):
        A[j][k] = A[j][k] - r*A[i][k]

  # Back Subsitution:
  s[len(s) - 1] = (A[len(A) - 1][len(A)])/(A[len(A)-1][len(A) -1])

  for i in range(len(A) - 2, -1, -1):
    s[i] = A[i][len(A)]
    for j in range(i+1, len(A)):
      s[i] = s[i] - A[i][j]*s[j]
    s[i] = (s[i])/(A[i][i])

  
  return s

# To get upper triangular matrix:s
def upperTri(n,A):
  # Test if the array is square:
  assert len(A) == len(A[0]) - 1

  # Solution Vector:
  s = np.zeros(len(A))

  # Gaussian Elimination:
  for i in range(len(A)):
    # Optimistically Assume a zero pivot is never encountered.
    for j in range(i+1 , n):
      r = (A[j][i])/(A[i][i])
      for k in range(n+1):
        A[j][k] = A[j][k] - r*A[i][k]
  # Back Subsitution:
  s[len(s) - 1] = (A[len(A) - 1][len(A)])/(A[len(A)-1][len(A) -1])

  for i in range(len(A) - 2, -1, -1):
    s[i] = A[i][len(A)]
    for j in range(i+1, len(A)):
      s[i] = s[i] - A[i][j]*s[j]
    s[i] = (s[i])/(A[i][i])

  return A



# n= 1000, part c) problem 2:

n = 1000
rng = default_rng()
x1000 = np.zeros(n)

for i in range(0,1000):
  x1000[i] = i+1
# Populate with values from the normal distribution:
A = rng.standard_normal((n,n))

b = A.dot(x1000)

augmentedA = np.hstack((A,np.atleast_2d(b).T))


timestart = time.time()
result = gaussElim(n,augmentedA)
print('Time for 1000 by 1000 Matrix:',result[0], "--- %s seconds ---" % (time.time() - timestart))

















