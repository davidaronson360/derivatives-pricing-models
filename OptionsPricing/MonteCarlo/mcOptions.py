#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 09:47:20 2025

@author: davidaronson
"""

# Monte Carlo European option pricer with variance reduction (antithetic + control variates)

GLOBAL_SEED = 2026  # Change this once to control all MC randomness

import numpy as np
import math
import matplotlib.pyplot as plt

# -----------------------------
# Core GBM path simulator
# -----------------------------
def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    antithetic: bool = False,
    seed: int | None = None,
    return_full_paths: bool = False
):
    """
    Simulate GBM paths under risk-neutral measure:
        dS = r S dt + sigma S dW
    Returns:
        - terminal prices (n_paths,)
        - optionally full paths shape (n_paths, n_steps+1) if return_full_paths=True
    """
    if antithetic and (n_paths % 2 == 1):
        n_paths += 1  # make even for pairing; we'll drop the last later

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    nudt = (r - 0.5 * sigma**2) * dt
    sigsdt = sigma * math.sqrt(dt)

    # Start all paths at S0
    if return_full_paths:
        paths = np.empty((n_paths, n_steps + 1), dtype=float)
        paths[:, 0] = S0
    else:
        paths = None

    # Generate random normals (with optional antithetic pairing)
    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal((half, n_steps))
        Z = np.vstack([Z, -Z])  # pair each draw with its negative
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    # Evolve paths in log-space for numerical stability
    logS = np.log(S0) + np.cumsum(nudt + sigsdt * Z, axis=1)
    S = np.exp(logS)

    if return_full_paths:
        paths[:, 1:] = S
    ST = S[:, -1]

    if antithetic and (paths is not None):
        # If we padded to even, truncate back to original requested size
        paths = paths[:n_paths]
    if antithetic:
        ST = ST[:n_paths]  # already correct; keep for symmetry

    return (ST, paths) if return_full_paths else (ST, None)


# -----------------------------
# European option payoff helpers
# -----------------------------
def payoff_call(ST, K):
    return np.maximum(ST - K, 0.0)

def payoff_put(ST, K):
    return np.maximum(K - ST, 0.0)

# -----------------------------
# Control variate using S_T (mean known: E[S_T]=S0*exp(rT))
# -----------------------------
def apply_control_variate(payoffs, ST, S0, r, T):
    target_mean = S0 * math.exp(r * T)
    cov = np.cov(payoffs, ST, ddof=1)[0, 1]
    var = np.var(ST, ddof=1)
    if var == 0:
        b_star = 0.0
    else:
        b_star = cov / var
    adjusted = payoffs - b_star * (ST - target_mean)
    return adjusted, b_star

# -----------------------------
# Monte Carlo pricer
# -----------------------------
def price_european_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 200_000,
    n_steps: int = 252,
    option_type: str = "call",
    antithetic: bool = True,
    control_variate: bool = True,
    seed: int | None = 42,
    return_ci: bool = True
):
    ST, _ = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, antithetic=antithetic, seed=seed)

    if option_type.lower() == "call":
        payoffs = payoff_call(ST, K)
    elif option_type.lower() == "put":
        payoffs = payoff_put(ST, K)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    b_star = None
    if control_variate:
        payoffs, b_star = apply_control_variate(payoffs, ST, S0, r, T)

    disc = math.exp(-r * T)
    disc_payoffs = disc * payoffs

    price = float(np.mean(disc_payoffs))
    std_err = float(np.std(disc_payoffs, ddof=1) / math.sqrt(len(disc_payoffs)))

    if not return_ci:
        return {"price": price, "stderr": std_err, "b_star": b_star}

    # 95% CI (normal approx)
    ci_low = price - 1.96 * std_err
    ci_high = price + 1.96 * std_err
    return {"price": price, "stderr": std_err, "ci_95": (ci_low, ci_high), "b_star": b_star}


# -----------------------------
# Black-Scholes closed form (for quick sanity checks)
# -----------------------------
def _norm_cdf(x):
    # standard normal CDF via error function
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def black_scholes_price(S0, K, T, r, sigma, option_type="call"):
    if T <= 0:
        if option_type == "call":
            return max(S0 - K, 0.0)
        else:
            return max(K - S0, 0.0)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S0 * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S0 * _norm_cdf(-d1)


# -----------------------------
# Demo run + quick convergence plot
# -----------------------------
S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
true_call = black_scholes_price(S0, K, T, r, sigma, "call")
true_put  = black_scholes_price(S0, K, T, r, sigma, "put")

res_call = price_european_mc(S0, K, T, r, sigma, n_paths=200_000, n_steps=252, option_type="call", antithetic=True, control_variate=True, seed=GLOBAL_SEED)
res_put  = price_european_mc(S0, K, T, r, sigma, n_paths=200_000, n_steps=252, option_type="put",  antithetic=True, control_variate=True, seed=GLOBAL_SEED)

print("Parameters: S0=%.2f K=%.2f T=%.2f r=%.2f sigma=%.2f" % (S0, K, T, r, sigma))
print(f"Black-Scholes  Call: {true_call:.6f}")
print(f"MC (AV+CV)     Call: {res_call['price']:.6f}  stderr={res_call['stderr']:.6f}  95%CI={res_call['ci_95']}  b*={res_call['b_star']:.4f}")
print(f"Black-Scholes   Put: {true_put:.6f}")
print(f"MC (AV+CV)      Put: {res_put['price']:.6f}  stderr={res_put['stderr']:.6f}  95%CI={res_put['ci_95']}  b*={res_put['b_star']:.4f}")

# Convergence vs number of paths (calls only, quick look)
path_grid = np.logspace(3, 5.5, num=10, base=10).astype(int)  # 1e3 .. ~3e5
estimates = []
errs = []
for n in path_grid:
    out = price_european_mc(S0, K, T, r, sigma, n_paths=n, n_steps=128, option_type="call", antithetic=True, control_variate=True, seed=GLOBAL_SEED)
    estimates.append(out["price"])
    errs.append(out["stderr"])

# Plot (no specific styles/colors per instructions)
plt.figure(figsize=(7,4.2))
plt.axhline(true_call, linestyle="--")
plt.plot(path_grid, estimates, marker="o")
plt.xscale("log")
plt.xlabel("Number of paths (log scale)")
plt.ylabel("MC price estimate")
plt.title("European Call MC Convergence (with antithetic + control variate)")
plt.tight_layout()
plt.show()

# Also plot stderr decay
plt.figure(figsize=(7,4.2))
plt.plot(path_grid, errs, marker="o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of paths (log scale)")
plt.ylabel("Std. Error (log scale)")
plt.title("Monte Carlo Standard Error vs Paths")
plt.tight_layout()
plt.show()
