#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 15:57:23 2025

@author: davidaronson
"""

import math
from scipy.stats import norm

def validate_inputs(S, K, T, r, sigma):
    if S <= 0:
        raise ValueError("Stock price S must be positive.")
    if K <= 0:
        raise ValueError("Strike price K must be positive.")
    if T <= 0:
        raise ValueError("Time to maturity T must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive.")

def d1_d2(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def call_price_bs(S, K, T, r, sigma):
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

def put_price_bs(S, K, T, r, sigma):
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def greeks_bs(S, K, T, r, sigma, option_type='call'):
    d1, d2 = d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * pdf_d1 * math.sqrt(T) / 100
    theta_call = (-S * pdf_d1 * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
    theta_put  = (-S * pdf_d1 * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
    rho_call = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
    rho_put  = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
    
    return {
        'Delta': round(delta, 4),
        'Gamma': round(gamma, 4),
        'Vega': round(vega, 4),
        'Theta': round(theta_call if option_type == 'call' else theta_put, 4),
        'Rho': round(rho_call if option_type == 'call' else rho_put, 4)
    }

def black_scholes(S, K, T, r, sigma):
    """
    Calculate European call and put option prices using the Black-Scholes formula.

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to maturity in years
    r (float): Risk-free interest rate (annual)
    sigma (float): Volatility of the underlying asset (annualized)

    Returns:
    dict: Dictionary containing 'Call Price', 'Put Price', and Greeks
    """
    validate_inputs(S, K, T, r, sigma)
    call = call_price_bs(S, K, T, r, sigma)
    put = put_price_bs(S, K, T, r, sigma)
    call_greeks = greeks_bs(S, K, T, r, sigma, option_type='call')
    put_greeks = greeks_bs(S, K, T, r, sigma, option_type='put')

    return {
        'Call Price': round(float(call), 2),
        'Put Price': round(float(put), 2),
        'Call Greeks': {k: round(float(v), 4) for k, v in call_greeks.items()},
        'Put Greeks': {k: round(float(v), 4) for k, v in put_greeks.items()}
    }


if __name__ == "__main__":
    S = 131
    K = 121
    T = 0.75  # time in years
    r = 0.05
    sigma = 0.2371

    result = black_scholes(S, K, T, r, sigma)
    
    print(f"Call Option Price: ${result['Call Price']:.2f}")
    print(f"Put Option Price : ${result['Put Price']:.2f}\n")
    
    print("Call Greeks:")
    for k, v in result['Call Greeks'].items():
        print(f"  {k}: {v}")
    
    print("\nPut Greeks:")
    for k, v in result['Put Greeks'].items():
        print(f"  {k}: {v}")

