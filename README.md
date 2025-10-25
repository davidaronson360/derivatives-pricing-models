# Derivatives Pricing & Stochastic Models
*Python implementations of core stochastic and option pricing models.*

This repository implements a suite of **derivatives pricing** and **stochastic process** models in Python.  
It serves as a conceptual and computational bridge between financial mathematics and quantitative modeling — translating theory into working, verifiable code.

--------------------------------------------------------------------------------
OVERVIEW
--------------------------------------------------------------------------------

The repository is organized into two main modules:

- **OptionsPricing/**
  - Closed-form and simulation-based option pricing models.
- **StochProcesses/**
  - Continuous-time stochastic processes used to model underlying asset dynamics.

Each folder contains a single, self-contained Python script illustrating the mathematics, simulation, and visualization of the corresponding model.

--------------------------------------------------------------------------------
IMPLEMENTED MODELS
--------------------------------------------------------------------------------

Black–Scholes  
    • bsm.py — Closed-form solution for European call/put options; includes analytical Greeks.

Brownian Motion  
    • brownianMotion.py — Simulates standard Brownian paths and visualizes variance growth over time.

Geometric Brownian Motion (GBM)  
    • geoBrownianMotion.py — Models log-normal price evolution under drift and volatility.

Heston Model  
    • hestonModel.py — Implements stochastic volatility dynamics with correlated Wiener processes.

Merton Jump Diffusion  
    • mertonJumpDiffusion.py — Extends GBM with Poisson-driven jumps in returns.

Ornstein–Uhlenbeck (OU)  
    • ouProcess.py — Mean-reverting process commonly used for spreads and interest rates.

Monte Carlo Option Pricing  
    • mcOptions.py — Estimates option prices via random path simulations and expected payoffs.

Implied Volatility Calculator  
    • impliedVolCalc.py - Numerically computes implied volatility.

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------

Python ≥ 3.9  
numpy  
pandas  
matplotlib  
scipy  

Install dependencies:
    pip install numpy pandas matplotlib scipy

