
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

# input data

d = [6,5,8,5,5,6,4,6,8,7,8,6,9,5,7,7,9,4,10,11,8,5,5,7,6,5,8,12,8,
    11,7,6,4,4,8,12,19,9,9,4,10,9,12,7,6,7,6,11,9,7,12,5,10,5,5,11,5,10,11,
    5,6,7,6,5,6,15,4,6,5,5,7,5,9,11,4,11,7,6,3,2,9,11,21,11,6,16,9,8,5,8,6,
    10,10,11,11,6,9,9,13,13,10,7,14,11,12,8,8,10,11,7,11,5,9,12,6,11,8,8,
    18,10,13,7,8,11,9,10,6,12,9,11,12,6,4,13,11,6,7,12,11,14,8,7,8,7,19,
    13,8,14,12,15,15,12,16,15,28,10,15,16,9,19,19,14,8] 

years = range(1851, 2014)

d_series = pd.Series(d,index=years)

# Plot the input data. Note this works best in iPython notebook
d_series.plot(title='Number of tropical storms in the North Atlantic per year')


# main changepoint detection algorithm
# single changepoint model

def step4(d):

    n = len(d)
    # dbar = sum(d)/float(n)
    dbar = np.mean(d)

    # dsbar = sum (d*d)/float(n)
    dsbar = np.mean(np.multiply(d,d))

    fac = dsbar-np.square(dbar)

    summ = 0
    summup = []

    for z in range(n):
        summ+=d[z]
        summup.append(summ)

    y = []

    for m in range(n-1):
        pos=m+1
        mscale = 4*(pos)*(n-pos)
        Q = summup[m]-(summ-summup[m])
        U = -np.square(dbar*(n-2*pos) + Q)/float(mscale) + fac
        y.append(-(n/float(2)-1)*math.log(n*U/2) - 0.5*math.log((pos*(n-pos))))

    z, zz = np.max(y), np.argmax(y)

    mean1 = sum(d[:zz+1])/float(len(d[:zz+1]))
    mean2=sum(d[(zz+1):n])/float(n-1-zz)

    return y, zz, mean1, mean2


# calling of main function

step_like = step4(d)

step_series = pd.Series(step_like[0],index=years[1:])


# Plot the result. Note this works best in iPython notebook
plt.figure(); 

step_series.plot(title='Log likelihood of step change location in number of tropical storms per year')

