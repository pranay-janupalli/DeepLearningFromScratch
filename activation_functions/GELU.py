#combines RELU and dropout
#Gausian Error Limit Unit
#https://youtu.be/kMpptn-6jaw?si=6U8tylnACqPnFrhv

"""
GELU(x)=x∗Φ(x)
where Φ(x) is the Cumulative Distribution Function for Gaussian Distribution.
When the approximate argument is tanh, Gelu is estimated with:
"""

import math 

def tanh(x):
    return (math.exp(x)-math.exp(-1*x))/(math.exp(x)+math.exp(-1*x))

def gelu(x):
    inner_value=math.sqrt(2/math.pi)*(x+0.044715*x**3)
    return 0.5*x*(1+tanh(inner_value))