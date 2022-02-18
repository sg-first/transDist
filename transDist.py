import numpy as np
import scipy.stats as stat
from scipy.optimize import fsolve

def getNormVal(mean, sigma2, size, sampleNum, newDistMaxX):
    result=[]
    toMovedX = lambda x: x - (newDistMaxX-mean)

    i=0
    while i<=size:
        iVal=toMovedX(i)
        cdfSub=stat.norm.cdf(iVal+0.5,loc=mean,scale=sigma2)-stat.norm.cdf(iVal-0.5,loc=mean,scale=sigma2)
        result.append(cdfSub*sampleNum)
        i+=1
    return np.array(result)

def toNorm(dist,newDistMaxX=None):
    size=len(dist)
    dist=np.array(dist)
    maxX=np.argmax(dist) # X轴向左平移多少
    toMovedX = lambda x: x - maxX
    sampleNum=np.sum(dist)
    # 均值
    mean=0
    for i in range(size):
        mean=toMovedX(i)*dist[i]
    mean/=sampleNum
    # 方差
    sigma2=0
    for i in range(size):
        sigma2+=dist[i]*(toMovedX(i)-mean)**2
    sigma2/=sampleNum

    if newDistMaxX is None:
        newDistMaxX=maxX
    return getNormVal(mean,sigma2,size,sampleNum,newDistMaxX)


def toUniform(dist2,newDistMaxX=None):
    n=len(dist2)
    dist2 = np.array(dist2)
    avg = np.average(dist2)
    var = np.var(dist2)

    def equ(i):
        x=i[0]
        y=i[1]
        return [(x + (n - 1) * y)/n - avg,
                ((((n - 1) * (y-avg)**2) + (x-avg)**2) / n) - var]

    r = fsolve(equ, np.array([0, 0]))

    if newDistMaxX is None:
        newDistMaxX=np.argmax(dist2)
    dist2_true_norm = np.full((n,), r[1])
    dist2_true_norm[newDistMaxX] = r[0]
    return dist2_true_norm

testList=[1,2,3,2,1]
print(toNorm(testList))