#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spiked_svm_bias.py

Section 11.4 — Modified bias term (Proposition 1).

Measures the intercept b in the u-scale and compares it with the predictions
from kappa and kappa_* in (11.2).

Usage:
    python spiked_svm_bias.py
"""

import numpy as np

from spiked_svm_common import make_sd, solve_hard_margin


def exp_bias(d, c1, c2, delta, n1, n2, n_rep, seed):
    """Measure intercept b and total dual weight A = sum_j alpha_1j."""
    rng = np.random.default_rng(seed)
    theta = float(d)
    sd1, _, _, _ = make_sd(d, [c1])
    sd2, _, _, _ = make_sd(d, [c2])
    Delta = delta * theta
    mu1 = np.zeros(d)
    mu1[0] = np.sqrt(Delta) / 2.0
    y = np.r_[np.ones(n1), -np.ones(n2)]
    bs = np.empty(n_rep)
    As = np.empty(n_rep)
    for r in range(n_rep):
        X1 = mu1 + rng.standard_normal((n1, d)) * sd1
        X2 = -mu1 + rng.standard_normal((n2, d)) * sd2
        U = np.vstack([X1, X2]) / np.sqrt(theta)
        _, b, a = solve_hard_margin(U @ U.T, y)
        bs[r] = b
        As[r] = a[:n1].sum()
    return (
        float(bs.mean()),
        float(bs.std(ddof=1) / np.sqrt(n_rep)),
        float(As.mean()),
    )


def main():
    d = 8000
    delta = 1.0
    n_rep = 300
    configs = [
        (5, 5, 0.2, 0.8),
        (4, 4, 0.1, 0.7),
        (10, 5, 0.4, 0.7),
        (8, 4, 0.3, 0.65),
        (8, 4, 0.5, 0.5),
    ]

    print(
        f"{'n1':>3}{'n2':>4}{'c1':>5}{'c2':>5}"
        f"{'k/th':>9}{'k*/th':>9}{'b':>12}{'se':>9}{'A':>8}"
        f"{'pred_k':>9}{'pred_k*':>9}"
    )
    print("-" * 90)
    for i, (n1, n2, c1, c2) in enumerate(configs):
        k = 1.0 / n1 - 1.0 / n2
        ks = (1.0 - c1) / n1 - (1.0 - c2) / n2
        b, se, A = exp_bias(d, c1, c2, delta, n1, n2, n_rep, seed=80 + i)
        pred_k = -0.5 * A * k
        pred_ks = -0.5 * A * ks
        print(
            f"{n1:>3}{n2:>4}{c1:>5.2f}{c2:>5.2f}"
            f"{k:>9.4f}{ks:>9.4f}{b:>12.4f}{se:>9.4f}{A:>8.3f}"
            f"{pred_k:>9.4f}{pred_ks:>9.4f}"
        )


if __name__ == "__main__":
    main()
