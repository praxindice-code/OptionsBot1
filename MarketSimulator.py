import random

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import vollib.black_scholes as bs
import vollib.black_scholes.greeks.analytical as greeks


def market_simulator(Volatility, ticksDesired, s0, trading_day_fraction=1/252):
    dt = trading_day_fraction / ticksDesired
    percentChanges = np.random.normal(loc=0, scale=Volatility * np.sqrt(dt), size=ticksDesired)
    priceArray = s0 * np.exp(np.cumsum(percentChanges[:ticksDesired]))
    return priceArray


def calculate_option_prices(priceArray, strike, time_to_expire, rate, Volatility, ticksDesired):
    optionPriceList = []
    original_time = time_to_expire
    for tick in range(ticksDesired):
        current_time = original_time * (1 - (tick / ticksDesired))
        current_time = max(current_time, 1e-6)
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


def compute_risk_covariance(price_array, warmup_ticks=None):
    if warmup_ticks is not None:
        path = price_array[:warmup_ticks]
    else:
        path = price_array
    dS = np.diff(path)
    dS_squared_half = 0.5 * dS**2
    factors = np.vstack([dS, dS_squared_half])
    Sigma = np.cov(factors)
    return Sigma


def linear_greeks_reservation_price(fair_value, net_delta, net_gamma, risk_aversion, sigma, time_remaining,
                                      delta_weight=1.0, gamma_weight=1.0):
    risk_measure = delta_weight * net_delta + gamma_weight * net_gamma
    return fair_value - risk_measure * risk_aversion * sigma**2 * time_remaining


def quadratic_reservation_price(fair_value, net_delta, net_gamma, Sigma, risk_aversion, time_remaining):
    g = np.array([net_delta, net_gamma])
    R = g @ Sigma @ g
    return fair_value - R * risk_aversion * time_remaining


def optimal_spread(gamma, sigma, time_remaining, k):
    return gamma * sigma**2 * time_remaining + (2 / gamma) * np.log(1 + gamma / k)


def run_simulation(
    n_ticks=1000,
    s0=90,
    strike=100,
    starting_cash=100,
    time_to_expire=1,
    rate=0.01,
    AssumedVolatility=0.2,
    gamma=0.1,
    k=1.5,
    Orders=140,
    FillRate=1.5,
    fill_dt=0,
    max_inventory=10,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    cash = starting_cash
    inventory = 0
    pnl_history = []

    priceArray = market_simulator(AssumedVolatility, n_ticks, s0=s0)
    optionPriceList = calculate_option_prices(
        priceArray, strike=strike, time_to_expire=time_to_expire,
        rate=rate, Volatility=AssumedVolatility, ticksDesired=n_ticks
    )

    for i in range(n_ticks):
        Price = priceArray[i]
        OptionFairValue = optionPriceList[i]

        reservationPrice = reservation_price(
            OptionFairValue, inventory, gamma=gamma,
            sigma=AssumedVolatility, time_remaining=(1 - i / n_ticks)
        )
        spread = optimal_spread(gamma=gamma, sigma=AssumedVolatility, time_remaining=(1 - i / n_ticks), k=k)

        bid = max(reservationPrice - spread / 2, 0.01)
        ask = reservationPrice + spread / 2

        can_buy = inventory < max_inventory
        can_sell = inventory > -max_inventory

        if can_buy and random.random() < fill_probability(abs(OptionFairValue - bid), Orders, FillRate, fill_dt):
            inventory += 1
            cash -= bid

        if can_sell and random.random() < fill_probability(abs(OptionFairValue - ask), Orders, FillRate, fill_dt):
            inventory -= 1
            cash += ask

        pnl = cash + inventory * OptionFairValue
        pnl_history.append(pnl)

    return {
        "final_pnl": pnl_history[-1],
        "final_inventory": inventory,
        "min_pnl": min(pnl_history),
        "max_pnl": max(pnl_history),
        "pnl_history": pnl_history
    }


def run_simulation_linear_greeks(
    n_ticks=1000,
    s0=90,
    strike=100,
    starting_cash=100,
    time_to_expire=1,
    rate=0.01,
    AssumedVolatility=0.2,
    risk_aversion=0.1,
    k=1.5,
    Orders=140,
    FillRate=1.5,
    fill_dt=0,
    max_inventory=10,
    delta_weight=1.0,
    gamma_weight=1.0,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    cash = starting_cash
    inventory = 0
    pnl_history = []

    priceArray = market_simulator(AssumedVolatility, n_ticks, s0=s0)
    optionPriceList = calculate_option_prices(
        priceArray, strike=strike, time_to_expire=time_to_expire,
        rate=rate, Volatility=AssumedVolatility, ticksDesired=n_ticks
    )

    for i in range(n_ticks):
        Price = priceArray[i]
        OptionFairValue = optionPriceList[i]
        time_remaining = 1 - i / n_ticks
        current_time = max(time_to_expire * time_remaining, 1e-6)

        delta_per_contract = greeks.delta('c', Price, strike, current_time, rate, AssumedVolatility)
        gamma_per_contract = greeks.gamma('c', Price, strike, current_time, rate, AssumedVolatility)
        net_delta = inventory * delta_per_contract
        net_gamma = inventory * gamma_per_contract

        reservationPrice = linear_greeks_reservation_price(
            OptionFairValue, net_delta, net_gamma, risk_aversion, AssumedVolatility, time_remaining,
            delta_weight=delta_weight, gamma_weight=gamma_weight
        )
        spread = optimal_spread(gamma=risk_aversion, sigma=AssumedVolatility, time_remaining=time_remaining, k=k)

        bid = max(reservationPrice - spread / 2, 0.01)
        ask = reservationPrice + spread / 2

        can_buy = inventory < max_inventory
        can_sell = inventory > -max_inventory

        if can_buy and random.random() < fill_probability(abs(OptionFairValue - bid), Orders, FillRate, fill_dt):
            inventory += 1
            cash -= bid

        if can_sell and random.random() < fill_probability(abs(OptionFairValue - ask), Orders, FillRate, fill_dt):
            inventory -= 1
            cash += ask

        pnl = cash + inventory * OptionFairValue
        pnl_history.append(pnl)

    return {
        "final_pnl": pnl_history[-1],
        "final_inventory": inventory,
        "min_pnl": min(pnl_history),
        "max_pnl": max(pnl_history),
        "pnl_history": pnl_history,
    }


def run_batch_linear_greeks(n_runs=100, **kwargs):
    results = []
    for i in range(n_runs):
        result = run_simulation_linear_greeks(seed=i, **kwargs)
        results.append(result)
    return results


def run_simulation_quadratic(
    n_ticks=1000,
    s0=90,
    strike=100,
    starting_cash=100,
    time_to_expire=1,
    rate=0.01,
    AssumedVolatility=0.2,
    risk_aversion=0.1,
    k=1.5,
    Orders=140,
    FillRate=1.5,
    fill_dt=0,
    max_inventory=10,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    cash = starting_cash
    inventory = 0
    pnl_history = []
    net_delta_history = []
    net_gamma_history = []

    priceArray = market_simulator(AssumedVolatility, n_ticks, s0=s0)
    optionPriceList = calculate_option_prices(
        priceArray, strike=strike, time_to_expire=time_to_expire,
        rate=rate, Volatility=AssumedVolatility, ticksDesired=n_ticks
    )

    calibration_path = market_simulator(AssumedVolatility, n_ticks, s0=s0,
                                          trading_day_fraction=1/252)
    Sigma = compute_risk_covariance(calibration_path)

    for i in range(n_ticks):
        Price = priceArray[i]
        OptionFairValue = optionPriceList[i]
        time_remaining = 1 - i / n_ticks
        current_time = max(time_to_expire * time_remaining, 1e-6)

        delta_per_contract = greeks.delta('c', Price, strike, current_time, rate, AssumedVolatility)
        gamma_per_contract = greeks.gamma('c', Price, strike, current_time, rate, AssumedVolatility)

        net_delta = inventory * delta_per_contract
        net_gamma = inventory * gamma_per_contract
        net_delta_history.append(net_delta)
        net_gamma_history.append(net_gamma)

        reservationPrice = quadratic_reservation_price(
            OptionFairValue, net_delta, net_gamma, Sigma, risk_aversion, time_remaining
        )
        spread = optimal_spread(gamma=risk_aversion, sigma=AssumedVolatility, time_remaining=time_remaining, k=k)

        bid = max(reservationPrice - spread / 2, 0.01)
        ask = reservationPrice + spread / 2

        can_buy = inventory < max_inventory
        can_sell = inventory > -max_inventory

        if can_buy and random.random() < fill_probability(abs(OptionFairValue - bid), Orders, FillRate, fill_dt):
            inventory += 1
            cash -= bid

        if can_sell and random.random() < fill_probability(abs(OptionFairValue - ask), Orders, FillRate, fill_dt):
            inventory -= 1
            cash += ask

        pnl = cash + inventory * OptionFairValue
        pnl_history.append(pnl)

    return {
        "final_pnl": pnl_history[-1],
        "final_inventory": inventory,
        "min_pnl": min(pnl_history),
        "max_pnl": max(pnl_history),
        "pnl_history": pnl_history,
        "net_delta_history": net_delta_history,
        "net_gamma_history": net_gamma_history,
    }


def run_batch_quadratic(n_runs=100, **kwargs):
    results = []
    for i in range(n_runs):
        result = run_simulation_quadratic(seed=i, **kwargs)
        results.append(result)
    return results


def run_batch(n_runs=100, **kwargs):
    results = []
    for i in range(n_runs):
        result = run_simulation(seed=i, **kwargs)
        results.append(result)
    return results


def summarize(results, label=""):
    final_pnls = [r["final_pnl"] for r in results]
    print(f"--- {label} ---")
    print("Average PnL:", np.mean(final_pnls))
    print("Worst PnL:", np.min(final_pnls))
    print("Best PnL:", np.max(final_pnls))
    print("Std deviation:", np.std(final_pnls))
    print()


def sweep(param_name, param_values, n_runs=100, **fixed_kwargs):
    all_results = {}
    for value in param_values:
        kwargs = dict(fixed_kwargs)
        kwargs[param_name] = value
        results = run_batch(n_runs=n_runs, **kwargs)
        summarize(results, label=f"{param_name} = {value}")
        all_results[value] = results
    return all_results


def compare_three_models(s0_values, time_to_expire_values, n_runs=50, **fixed_kwargs):
    rows = []
    for s0 in s0_values:
        for tte in time_to_expire_values:
            kwargs = dict(fixed_kwargs)
            kwargs["s0"] = s0
            kwargs["time_to_expire"] = tte

            linear = run_batch(n_runs=n_runs, **kwargs)
            lin_greeks = run_batch_linear_greeks(n_runs=n_runs, **kwargs)
            quad = run_batch_quadratic(n_runs=n_runs, **kwargs)

            linear_pnls = [r["final_pnl"] for r in linear]
            lg_pnls = [r["final_pnl"] for r in lin_greeks]
            quad_pnls = [r["final_pnl"] for r in quad]

            rows.append({
                "s0": s0, "time_to_expire": tte,
                "linear_avg": np.mean(linear_pnls), "linear_worst": np.min(linear_pnls),
                "lin_greeks_avg": np.mean(lg_pnls), "lin_greeks_worst": np.min(lg_pnls),
                "quad_avg": np.mean(quad_pnls), "quad_worst": np.min(quad_pnls),
            })
            print(f"s0={s0}, T={tte}: "
                  f"linear worst={np.min(linear_pnls):.2f}, "
                  f"lin+greeks worst={np.min(lg_pnls):.2f}, "
                  f"quadratic worst={np.min(quad_pnls):.2f}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    comparison = compare_three_models(
        s0_values=[90, 100, 110],
        time_to_expire_values=[1, 0.1, 0.02],
        n_runs=50,
        max_inventory=10,
    )
    comparison.to_csv("three_model_comparison.csv", index=False)
    print("\nFull comparison table:")
    print(comparison.to_string(index=False))