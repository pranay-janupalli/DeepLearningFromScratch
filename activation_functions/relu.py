def relu(num):
    return max(0,num)

def leaky_relu(num,a=0.001):
    return max(a*num,num)
