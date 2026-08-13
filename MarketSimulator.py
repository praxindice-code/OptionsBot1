import numpy as np 
import matplotlib.pyplot as plt
import vollib.black_scholes as bs

n_ticks = 1000 
vol = .02

percentChanges = np.random.normal(loc=0, scale=.02, size=n_ticks) 

ticksDesired = 1000
s0 = 100
priceArray = s0 * np.exp(np.cumsum(percentChanges[:ticksDesired]))
#exp adds one 2 each array, cumsum combines each next element with previous element sum

strike = 150 
time_to_expire = .1
rate = .02
Volatility = .2
optionPriceList = []

for tick in range(ticksDesired):
    time_to_expire = time_to_expire - (tick / ticksDesired) * time_to_expire
    
    option_price = bs.black_scholes('c', priceArray[tick], strike, time_to_expire, rate, Volatility)
    optionPriceList.append(option_price) 
    

plt.plot(priceArray, label = "Stock Price")
plt.plot(optionPriceList, label = "Option Price")
plt.title("Option Price vs Stock Price")
plt.legend()
plt.show()