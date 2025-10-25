#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 09:25:41 2025

@author: davidaronson
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 3.0
N = 1000
dt = T / N
t = np.linspace(0, T, N + 1)

# Heston parameters
S0 = 100
v0 = 0.04
r = 0.05          # risk-free rate (use r for pricing)
kappa = 3.0
theta = 0.04
xi = 0.5
rho = -0.7
num_paths = 5

# RNG
rng = np.random.default_rng(42)

# Storage
S = np.full((num_paths, N + 1), S0, dtype=float)
v = np.full((num_paths, N + 1), v0, dtype=float)

# Pre-generate correlated normals
Z1 = rng.normal(size=(num_paths, N))
Z2 = rng.normal(size=(num_paths, N))
Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
dW_S = np.sqrt(dt) * Z1
dW_v = np.sqrt(dt) * Z2

# Simulate
for i in range(num_paths):
    for n in range(N):
        v_prev = v[i, n]
        v_clamped = v_prev if v_prev > 0 else 0.0  # full truncation
        # variance update (full truncation Euler)
        v_next = v_prev + kappa*(theta - v_clamped)*dt + xi*np.sqrt(v_clamped)*dW_v[i, n]
        v[i, n+1] = v_next if v_next > 0 else 0.0
        # asset update (log-Euler under Q)
        S[i, n+1] = S[i, n] * np.exp((r - 0.5 * v_clamped) * dt + np.sqrt(v_clamped) * dW_S[i, n])

# Plot S
plt.figure(figsize=(12, 5))
for i in range(num_paths):
    plt.plot(t, S[i], label=f'Path {i+1}' if i < 3 else None)
plt.title('Heston Model: Asset Price Paths')
plt.xlabel('Time')
plt.ylabel('S(t)')
plt.grid(True)
plt.legend()
plt.show()

# Plot v
plt.figure(figsize=(12, 4))
for i in range(num_paths):
    plt.plot(t, v[i], lw=1, alpha=0.85)
plt.title('Heston Model: Variance Paths (v)')
plt.xlabel('Time')
plt.ylabel('v(t)')
plt.grid(True)
plt.show()
