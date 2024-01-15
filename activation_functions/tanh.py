import math 

def tanh(x):
    return (math.exp(x)-math.exp(-1*x))/(math.exp(x)+math.exp(-1*x))
