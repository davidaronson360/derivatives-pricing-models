#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 15:40:27 2025

@author: davidaronson
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 10                 # total time
N = T * 10**2          # number of steps
dt = T / N             # time step size
mu = 0.1               # drift (e.g., 10% annual return)
sigma = 0.32            # volatility (e.g., 20% annual std dev)
S0 = 100               # initial value for GBM
num_paths = 100         # number of paths

# Time grid
t = np.linspace(0, T, N)

# Brownian increments (num_paths x N)
dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=(num_paths, N))

# Standard Brownian motion paths
W = np.cumsum(dW, axis=1)

# Prepend zero to each W path so W(0) = 0
W = np.hstack((np.zeros((num_paths, 1)), W))  # shape: (num_paths, N+1)

# Adjust time grid accordingly
t = np.linspace(0, T, N + 1)

# Geometric Brownian Motion paths
drift = (mu - 0.5 * sigma**2) * t
diffusion = sigma * W
S = S0 * np.exp(drift[np.newaxis, :] + diffusion)

# Plot GBM paths
plt.figure(figsize=(12, 5))
for i in range(num_paths):
    plt.plot(t, S[i], lw=1, alpha=0.8, label=f'Path {i+1}' if i < 5 else None)

plt.title(f'{num_paths} Simulated Geometric Brownian Motion Paths\nμ={mu}, σ={sigma}, S₀={S0}')
plt.xlabel('Time')
plt.ylabel('S(t)')
plt.grid(True)
plt.legend()
plt.show()
