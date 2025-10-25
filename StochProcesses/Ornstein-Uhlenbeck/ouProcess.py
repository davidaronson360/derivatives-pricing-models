#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 15:54:23 2025

@author: davidaronson
"""

import numpy as np
import matplotlib.pyplot as plt

# === Parameters ===
theta = 0.7     # rate of mean reversion
mu = 10.0        # long-term mean
sigma = 1     # volatility
X0 = 0.0        # initial value
T = 20.0        # total time
dt = 0.01       # time step
N = int(T / dt) # number of steps
n_paths = 100     # number of paths to simulate

# === Time and Noise ===
t = np.linspace(0, T, N)
dW = np.random.normal(scale=np.sqrt(dt), size=(n_paths, N))

# === Initialize ===
X = np.zeros((n_paths, N))
X[:, 0] = X0

# === Simulation (Euler-Maruyama) ===
for i in range(1, N):
    X[:, i] = X[:, i-1] + theta * (mu - X[:, i-1]) * dt + sigma * dW[:, i-1]

# === Plot ===
plt.figure(figsize=(10, 5))
for j in range(n_paths):
    plt.plot(t, X[j], lw=1)
plt.title(f'Ornstein-Uhlenbeck Process ({n_paths} paths)')
plt.xlabel('Time')
plt.ylabel('X(t)')
plt.grid(True)
plt.show()
