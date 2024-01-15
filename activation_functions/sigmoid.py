import math
def sigmoid(num):
    result=1/(1+math.exp(-1*num))
    return result
