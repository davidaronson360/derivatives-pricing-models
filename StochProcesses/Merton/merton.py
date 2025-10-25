#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 14:37:50 2025

@author: davidaronson
"""

import numpy as np
import matplotlib.pyplot as plt

def merton_jump_diffusion(S0, mu, sigma, lamb, muJ, sigmaJ, T, N, n_paths=10):
    dt = T / N
    kappa = np.exp(muJ + 0.5 * sigmaJ**2) - 1

    # Initialize paths
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = S0

    for i in range(1, N + 1):
        # Brownian motion part
        Z = np.random.normal(size=n_paths)
        dW = sigma * np.sqrt(dt) * Z
        
        # Poisson jump part
        N_jumps = np.random.poisson(lamb * dt, size=n_paths)
        J = np.random.normal(muJ, sigmaJ, size=n_paths)  # log-normal jump
        jumps = np.exp(J) - 1

        # Apply update
        paths[:, i] = paths[:, i-1] * np.exp((mu - lamb * kappa) * dt + dW) * (1 + jumps * N_jumps)

    return paths

# Parameters
S0 = 100         # initial price
mu = 0.1         # drift
sigma = 0.2      # volatility
lamb = 0.75      # jump intensity (lambda)
muJ = -0.1       # mean of jump size (log-normal)
sigmaJ = 0.3     # std of jump size
T = 1            # time horizon
N = 252          # steps (e.g. trading days)
n_paths = 10     # number of paths

# Simulate paths
paths = merton_jump_diffusion(S0, mu, sigma, lamb, muJ, sigmaJ, T, N, n_paths)

# Plot
t = np.linspace(0, T, N+1)
plt.figure(figsize=(12, 5))
for i in range(n_paths):
    plt.plot(t, paths[i], lw=1)
plt.title('Merton Jump Diffusion Paths')
plt.xlabel('Time')
plt.ylabel('Asset Price')
plt.grid(True)
plt.show()
