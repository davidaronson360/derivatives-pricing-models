#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 15:13:27 2025

@author: davidaronson
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 10           # total time
N = T*10**2     # number of steps
dt = T / N      # time step size
mu = 0          # drift
sigma = 1       # volatility
num_paths = 10  # Number of Brownian motion paths


# Time grid
t = np.linspace(0, T, N)

# Brownian increments: shape (num_paths, N)
dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=(num_paths, N))

# Brownian paths: cumulative sum along axis 1 (time axis)
W = np.cumsum(dW, axis=1)

# Plot
plt.figure(figsize=(12, 5))
for i in range(num_paths):
    plt.plot(t, W[i], lw=1, alpha=0.8, label=f'Path {i+1}' if i < 5 else None)

plt.title(f'{num_paths} Simulated Brownian Motion Paths')
plt.xlabel('Time')
plt.ylabel('W(t)')
plt.grid(True)
plt.legend()
plt.show()