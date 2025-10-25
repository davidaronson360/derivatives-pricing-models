#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 14:40:00 2025

@author: davidaronson
"""

import scipy.stats as si
import numpy as np
from scipy.optimize import brentq

# Black-Scholes formula
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * si.norm.cdf(d1) - K * np.exp(-r*T) * si.norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r*T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price

# Objective function: difference between model and market price
def implied_vol_obj(sigma, S, K, T, r, market_price, option_type):
    return black_scholes_price(S, K, T, r, sigma, option_type) - market_price

# Implied volatility calculator
def implied_volatility(S, K, T, r, market_price, option_type='call'):
    try:
        return brentq(
            implied_vol_obj,
            a=1e-6,
            b=5.0,
            args=(S, K, T, r, market_price, option_type),
            maxiter=1000
        )
    except ValueError:
        return np.nan  # could not find a solution

# Example usage
S = 131       # current stock price
K = 121       # strike price
T = 0.75         # time to maturity in years
r = 0.05      # risk-free rate
market_price = 4.48  # observed market price
option_type = 'put'

iv = implied_volatility(S, K, T, r, market_price, option_type)
print(f"Implied Volatility: {iv:.4f}")
