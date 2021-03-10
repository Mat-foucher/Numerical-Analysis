import math
import numpy as np
from scipy.integrate import quad
from scipy.linalg import hilbert
import time

start_time = time.time()
# Problem 2a) Code the program to evaulate multidimensional integrals.


# def test1(x,y):
#     return 1/(x + y)

# def test2(x,y,z):
#     return 1/(x + y + z)

# def test3(x,y,z,w):
#     return 1/(x+y+z+w)

# Double Integral 
def int2dim(f,bds):
    assert len(bds) == 4
    fy = lambda y: quad(f,bds[0],bds[1],args=(y))[0]
    value,err = quad(fy,bds[2],bds[3])
    return value, err



# Triple Integral
def int3dim(f,bds):
    assert len(bds) == 6
    fyz = lambda y,z: quad(f,bds[0],bds[1],args=(y,z))[0]
    fz = lambda z: quad(fyz,bds[2],bds[3],args=(z))[0]
    value,err = quad(fz, bds[4],bds[5])
    return value,err

# Quadruple
def int4dim(f,bds):
    assert len(bds) == 8
    fxyz = lambda x,y,z: quad(f,bds[0],bds[1],args=(x,y,z))[0]
    fyz = lambda y,z: quad(fxyz, bds[2],bds[3],args=(y,z))[0]
    fz = lambda z: quad(fyz, bds[4],bds[5],args=(z))[0]
    value, err = quad(fz,bds[6],bds[7])
    return value, err




# The function will take the bounds in as a list of numbers and calcuate the correct integral
def multiInt(function, bounds):
    # Single Variable:
    if len(bounds) == 2:
        value, err = quad(function,bounds[0],bounds[1])
        return value, err
    elif len(bounds) == 4:
        # Double Integral:
        value, err = int2dim(function,bounds)
        return value, err
    elif len(bounds) == 6:
        value, err = int3dim(function,bounds)
        return value, err
    elif len(bounds) == 8:
        value, err = int4dim(function, bounds)
        return value, err
    else:
        print('Invalid Number of bounds')
        return -1.0, -1.0

# Test (works good):
# bd1 = [1,2,1,2]

# v, e = multiInt(test1, bd1)

#print('Value 2d:',v, 'Error:',e)



def dimD(tuple_, dimension, hilb):

    def F(vector,hilb):
        #f = hilbert(dim).dot(np.transpose(vector))
        #f = vector.dot(f)
        return vector @ hilb @ vector.T

    """
    def P(vector, dim):
        sum = vector @ vector.T
        sum = -sum/2
        
        p = (1/((2*math.pi)**(dim/2)))*np.exp(sum)
        return p
    """
    return F(tuple_,hilb)
    

# def dimD(tuple_, Hilb):
#     def F(vector):
# 	    return vector @ Hilb @ np.transpose(vector)  # @ is shorthand for dot/matrix-multiply
# 		# vector.T is shorthand for np.transpose(vector)

#     def P(vector):
# 		sum = vector @ vector.T
# 		sum = -sum/2
#         p = (1/(2*math.pi)**(dim/2))*np.exp(sum)
#         return p

#     return F(tuple_)*P(tuple_)
"""
def dimension1(x):
    return dimD(np.array([x]), 1)
def dimension2(x,y):
    return dimD(np.array([x,y]),2)
def dimension3(x,y,z):
    return dimD(np.array([x,y,z]),3)
def dimension4(x,y,z,w):
    return dimD(np.array([x,y,z,w]),4)
"""
# IMPORTANT! DO NOT DELETE ANYTHING BELOW THIS LINE

# vd1,ed1 = multiInt(dimension1,[-5,5])
# vd2,ed2 = multiInt(dimension2, [-5,5,-5,5])
# vd3,ed3 = multiInt(dimension3, [-5,5,-5,5,-5,5])
# #vd4,ed4 = multiInt(dimension4, [-5,5,-5,5,-5,5,-5,5])

# print('Dimension 1:',vd1,'Error:',ed1)

# print('Dimension 2:',vd2, 'Error:',ed2)

# print('Dimension 3:', vd3,'Error:',ed3)

#print('Dimension 4:', vd4, 'Error:',ed4)

# Runtime of code, for the homework I will take this 
# and paste it into individual cells for the dimensions.
#print("--- %s seconds ---" % (time.time() - start_time))

