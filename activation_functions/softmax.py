import math

def softmax(arr):
    exps=list(map(math.exp,arr))
    sum_exps=sum(exps)
    for i in range(len(arr)):
        exps[i]=exps[i]/sum_exps
    return exps
