import numpy as np 
import matplotlib.pyplot as plt

n_ticks = 1000 
Volatility = .02 

percentChanges = np.random.normal(loc=0, scale=Volatility, size=n_ticks) 

ticksDesired = 1000
s0 = 100
priceArray = s0 * np.exp(np.cumsum(percentChanges[:ticksDesired]))
#exp adds one 2 each array, cumsum combines each next element with previous element sum



plt.plot(priceArray)
plt.xlabel("time")
plt.ylabel("price")
plt.show()

