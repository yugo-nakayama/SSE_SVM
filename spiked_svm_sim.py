#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spiked_svm_sim.py

Numerical verification of Theorem 4 (the limiting misclassification rate of the
hard-margin SVM under a single common spike).

Setting (E):
    n1 = n2 = 1, theta1 = theta2 = theta, m1 = m2 = 1,
    common spiked direction h, lambda = c * theta, h orthogonal to mu,
    non-spiked part tr(Sigma_{i*}) = (1-c) * theta, z standard normal.

Under this setting, for x0 in pi_1, the discriminant function satisfies

    y(x0)/theta   -->   delta/2 + c (z1 - z2) ( z0 - (z1 + z2)/2 )                 ... (7.1)

in distribution, and with A := c (z1 - z2) the limiting conditional
misclassification rate is

    e*(1) = Phi( sgn(A) * (z1 + z2)/2     -    delta / (2 |A|) )                   ... (7.2)

This script compares the Monte Carlo integral of (7.2) with the finite-dimensional
simulation of E{e(1)} for d = 20000.

Usage:
    python spiked_svm_sim.py
"""

import numpy as np
from scipy.stats import norm


# ----------------------------------------------------------------------
# 1. Monte Carlo integration of the limiting rate (7.2)
# ----------------------------------------------------------------------
def limit_error_rate(c, delta, n_mc=200_000, seed=1):
    """Compute E{e*(1)} in Theorem 4 from (7.2).

    Parameters
    ----------
    c : float
        Spike ratio lambda / theta in (0, 1).
    delta : float
        Signal ratio Delta / theta = ||mu1 - mu2||^2 / theta.
    n_mc : int
        Number of Monte Carlo points for the integration over (z1, z2).
    """
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n_mc)
    z2 = rng.standard_normal(n_mc)

    A = c * (z1 - z2)              # slope of the discriminant function in z0
    # (7.2): sgn(A) * (z1+z2)/2 - delta / (2|A|)
    arg = np.sign(A) * (z1 + z2) / 2.0 - delta / (2.0 * np.abs(A))
    return float(norm.cdf(arg).mean())   # integration over z0 is closed by Phi


# ----------------------------------------------------------------------
# 2. Finite-dimensional simulation
# ----------------------------------------------------------------------
def finite_d_error_rate(c, delta, d=20_000, n_rep=600, seed=2):
    """Estimate E{e(1)} by constructing the hard-margin SVM with n1 = n2 = 1.

    With one observation in each class, the hard-margin SVM is the perpendicular
    bisector of the two points, so that
        w = x11 - x21,   b = -w^T (x11 + x21) / 2
    as in (7.3).

    Choice of the coordinates:
        coordinate 0      : direction of the mean difference mu
        coordinate 1      : spiked direction h, orthogonal to mu
        coordinates 2..d-1: isotropic non-spiked part
    Then Sigma_i = diag(sig2, lambda, sig2, ..., sig2), and sig2 is chosen so
    that tr(Sigma_{i*}) = (d-1) * sig2 is approximately (1-c) * theta.
    """
    rng = np.random.default_rng(seed)

    theta = float(d)
    lam = c * theta                        # spiked eigenvalue
    sig2 = (1.0 - c) * theta / (d - 1)     # variance of each non-spiked coordinate
    Delta = delta * theta

    def draw(sign):
        """Generate one observation from pi_1 (sign=+1) or pi_2 (sign=-1)."""
        x = rng.normal(0.0, np.sqrt(sig2), d)   # non-spiked part
        x[0] += sign * np.sqrt(Delta) / 2.0     # mean difference
        x[1] = rng.normal(0.0, np.sqrt(lam))    # overwrite the spiked coordinate
        return x

    mu0 = np.zeros(d)                      # mean vector of x0 in pi_1
    mu0[0] = np.sqrt(Delta) / 2.0

    errs = np.empty(n_rep)
    for r in range(n_rep):
        x11 = draw(+1)
        x21 = draw(-1)

        w = x11 - x21
        b = -w @ (x11 + x21) / 2.0

        # Conditional mean and variance of f(x0) = w^T x0 + b given (w, b),
        # where x0 ~ N(mu0, Sigma_1); the integration over x0 is analytic.
        mean = w @ mu0 + b
        var = sig2 * np.sum(w ** 2) + (lam - sig2) * w[1] ** 2
        #       ^ all coordinates as sig2    ^ correction for coordinate 1

        errs[r] = norm.cdf(-mean / np.sqrt(var))    # P{f(x0) < 0}

    return float(errs.mean())


# ----------------------------------------------------------------------
# 3. Output of the comparison table (Table 2 in Section 11)
# ----------------------------------------------------------------------
def main():
    c_grid = [0.1, 0.3, 0.5, 0.8]
    delta_grid = [0.0, 1.0, 4.0]

    header = f"{'c':>5}{'delta':>8}{'limit (7.2)':>14}{'sim d=20000':>14}"
    print(header)
    print("-" * len(header))
    for c in c_grid:
        for delta in delta_grid:
            lim = limit_error_rate(c, delta)
            sim = finite_d_error_rate(c, delta)
            print(f"{c:>5}{delta:>8}{lim:>14.4f}{sim:>14.4f}")


if __name__ == "__main__":
    main()
