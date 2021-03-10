import math
import numpy as np
from scipy.integrate import quad
from scipy.linalg import hilbert
import random
import time
from problem2 import dimD

# Monte Carlo stuff:


# def dimD(tuple_, Hilb):
#     def F(vector):
# 	    return vector @ Hilb @ np.transpose(vector)  # @ is shorthand for dot/matrix-multiply
# 		# vector.T is shorthand for np.transpose(vector)

#     def P(vector):
# 	    sum = vector @ vector.T
# 		sum = -sum/2
#         p = (1/(2*math.pi)**(dim/2))*np.exp(sum)
#         return p

#     return F(tuple_)*P(tuple_)

"""
def monteCarlo(dimension, numPoints):
    assert numPoints > 0
    
    sum = 0.0
    
    x = np.random.random_sample((numPoints, dimension))
    
    Hilb = hilbert(dimension)
    for i in range(numPoints):
        sum = sum + dimD(x[i],Hilb)
        
    

    return (sum/numPoints)
"""

def F(vector,hilb):
        
    return vector.T @ hilb @ vector

def monteCarlo(dimension, numPoints):
    assert numPoints > 0
    
    #sum = [0]*numPoints
    value = 0.0
    j = 0
    hilb = hilbert(dimension)
    while j< numPoints:
        x = np.random.randn(dimension)
        
        

        fv = F(x, hilb)
        value = value + fv
        #sum[j] = fv
        j+=1
    #total = np.mean(sum)

    return value/numPoints


timestart = time.time()
s = monteCarlo(1,10**6)
print('Estimate for d=1',s, "--- %s seconds ---" % (time.time() - timestart))

timestart2 = time.time()
s2 = monteCarlo(2,10**6)
print('Estimate for d=2',s2, "--- %s seconds ---" % (time.time() - timestart2))

timestart3 = time.time()
s3 = monteCarlo(3,10**6)
print('Estimate for d=3',s3, "--- %s seconds ---" % (time.time() - timestart3))

timestart4 = time.time()
s4 = monteCarlo(4,10**6)
print('Estimate for d=4',s4, "--- %s seconds ---" % (time.time() - timestart4))

timestart12 = time.time()
s12 = monteCarlo(12,10**6)
print('Estimate for d=12',s12, "--- %s seconds ---" % (time.time() - timestart12))

timestart100 = time.time()
s100 = monteCarlo(100,10**6)
print('Estimate for d=100',s100, "--- %s seconds ---" % (time.time() - timestart100))
