import numpy as np
import scipy.stats as stat

def getNormVal(mean, sigma2, toMovedX, size, sampleNum):
    result=[]

    lastVal=toMovedX(0)
    i=1
    while i<=size:
        iVal=toMovedX(i)
        print(iVal)
        cdfSub=stat.norm.cdf(iVal,loc=mean,scale=sigma2)-stat.norm.cdf(lastVal,loc=mean,scale=sigma2)
        result.append(cdfSub*sampleNum)
        lastVal=iVal
        i+=1
    return result

def toNorm(dist):
    size=len(dist)
    dist=np.array(dist)
    maxX=np.argmax(dist) # X轴向左平移多少
    toMovedX=lambda x:x-maxX
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

    return getNormVal(mean,sigma2,toMovedX,size,sampleNum)

testList=[1,2,3,2,1]
print(toNorm(testList))