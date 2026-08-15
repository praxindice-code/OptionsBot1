import numpy as np 
import matplotlib.pyplot as plt
import vollib.black_scholes as bs

n_ticks = 1000 
vol = .02

percentChanges = np.random.normal(loc=0, scale=.02, size=n_ticks) 

ticksDesired = 1000
s0 = 100
priceArray = s0 * np.exp(np.cumsum(percentChanges[:ticksDesired]))
# cumsum combines each next element with previous element sum

strike = 150 
time_to_expire = .1
rate = .02
Volatility = .2
optionPriceList = []

for tick in range(ticksDesired):
    time_to_expire = time_to_expire - (tick / ticksDesired) * time_to_expire
    
    option_price = bs.black_scholes('c', priceArray[tick], strike, time_to_expire, rate, Volatility)
    optionPriceList.append(option_price)   

def fill_probability(distance, Orders=140, FillRate=1.5 ):
    OrderIntensity = Orders * np.exp(-distance * FillRate) 
    Probability = 1 - np.exp(-OrderIntensity * FillRate)

def reservation_price(fair_value, inventory, gamma, sigma, time_remaining):
    return fair_value - inventory * gamma * sigma**2 * time_remaining

def optimal_spread(gamma, sigma, time_remaining, k):
    return gamma * sigma**2 * time_remaining + (2 / gamma) * np.log(1 + gamma / k)