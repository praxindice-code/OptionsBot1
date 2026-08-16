import random

import numpy as np 
import matplotlib.pyplot as plt
import vollib.black_scholes as bs

AssumedVolatility = 0.2


optionPriceList = []
def market_simulator(Volatility, ticksDesired,s0,trading_day_fraction = 1/252):
    dt = trading_day_fraction / ticksDesired
    percentChanges = np.random.normal(loc=0, scale=Volatility * np.sqrt(dt), size=ticksDesired)
    
    priceArray = s0 * np.exp(np.cumsum(percentChanges[:ticksDesired]))
    return priceArray
    

    

def calculate_option_prices(priceArray, strike, time_to_expire, rate, Volatility, ticksDesired):
    original_time = time_to_expire
    for tick in range(ticksDesired):
        current_time = original_time  * (1- (tick / ticksDesired) )
    
        option_price = bs.black_scholes('c', priceArray[tick], strike, current_time, rate, Volatility)
        optionPriceList.append(option_price)
    return optionPriceList   

def fill_probability(distance, Orders=140, FillRate=1.5, dt=0):
    if dt == 0:
        dt = .001
    OrderIntensity = Orders * np.exp(-distance * FillRate) 
    Probability = 1 - np.exp(-OrderIntensity * dt)
    return Probability

def reservation_price(fair_value, inventory, gamma, sigma, time_remaining):
    return fair_value - inventory * gamma * sigma**2 * time_remaining

def optimal_spread(gamma, sigma, time_remaining, k):
    return gamma * sigma**2 * time_remaining + (2 / gamma) * np.log(1 + gamma / k)


pnl = []
cash = 100
inventory = 0
n_ticks = 1000
priceArray = market_simulator(AssumedVolatility, n_ticks, s0=90)
optionPriceList = calculate_option_prices(priceArray, strike=100, time_to_expire=1, rate=0.01, Volatility=AssumedVolatility, ticksDesired=n_ticks)
position_limit = 10

for i in range(n_ticks):
    can_buy = inventory > position_limit
    can_sell = inventory < -position_limit

    Price = priceArray[i]
    OptionFairValue = optionPriceList[i]
    
    reservationPrice = reservation_price(OptionFairValue, inventory=inventory, gamma=0.1, sigma=AssumedVolatility, time_remaining=(1 - i/n_ticks))
    spread = optimal_spread(gamma=0.1, sigma=AssumedVolatility, time_remaining=(1 - i/n_ticks), k=1.5)
    
    bid = max(reservationPrice - spread / 2, 0)
    ask = max(reservationPrice + spread / 2, 0)
    
    if can_buy and random.random() < fill_probability(abs(OptionFairValue - bid)):
        inventory += 1
        cash -= bid
        print(f"Filled bid at {bid} for price {Price}")
    if random.random() < fill_probability(abs(OptionFairValue - ask)):
        inventory -= 1
        cash += ask
        print(f"Filled ask at {ask} for price {Price}")
        
    pnl.append(cash + inventory * OptionFairValue)
print(f"Final PnL: {pnl[-1]}, Final Cash: {cash}, Final Inventory: {inventory}")