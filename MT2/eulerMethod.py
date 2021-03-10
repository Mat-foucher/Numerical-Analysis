import numpy as np
import math 
import scipy 
import time 
import matplotlib.pyplot as plt

def f(t,y):
    return (1+t)/(1+y)

# True Solution:
def y(t):
  return np.sqrt(t**2 + 2*t + 6) - 1

# Range for h-values:
hRange = np.logspace(-4,-14,10,base=10)

# Value is the t value, weight is the y value:
def eulerMethod(weight,svalue, evalue, func,step):
    
    t = np.arange(svalue,evalue,step)
    y = [0]*len(t)
    y[0] = weight
    for i in range(1, len(t)):
        y[i] = y[i-1] + func(svalue,y[i-1])*step
        svalue+= step
    return y
t1 = time.time()

yl = eulerMethod(2.0, 1.0, 1.0 + 10**(-5),f,10**(-7))



print(yl[len(yl)-1])


# Store Error:
errors = [0]*10




print('Time',t1, "--- %s seconds ---" % (time.time() - t1))

plt.figure(figsize=(8,6))
plt.plot(yl)
plt.show()