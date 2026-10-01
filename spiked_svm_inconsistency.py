#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spiked_svm_inconsistency.py

Section 11.3 — Inconsistency of the hard-margin SVM (Corollary 1).

(A) Average e(1)+e(2) vs d for fixed n1 = n2 = 2, delta = 0.25
    (Figure 1 / dropped Table 4).
(B) Average e(1)+e(2) vs n for fixed d = 4000 (Table n-vary).

Usage:
    python spiked_svm_inconsistency.py
"""

import numpy as np

from spiked_svm_common import exact_error, make_sd, solve_hard_margin


def exp_error(d, c, delta, n1, n2, n_rep, seed):
    """Average of e(1)+e(2) for the hard-margin SVM."""
    rng = np.random.default_rng(seed)
    theta = float(d)
    sd, var, _, _ = make_sd(d, [c])
    Delta = delta * theta
    mu1 = np.zeros(d)
    mu1[0] = np.sqrt(Delta) / 2.0
    y = np.r_[np.ones(n1), -np.ones(n2)]
    mu1u, mu2u = mu1 / np.sqrt(theta), -mu1 / np.sqrt(theta)
    out = np.empty(n_rep)
    for r in range(n_rep):
        X1 = mu1 + rng.standard_normal((n1, d)) * sd
        X2 = -mu1 + rng.standard_normal((n2, d)) * sd
        U = np.vstack([X1, X2]) / np.sqrt(theta)
        beta, b, _ = solve_hard_margin(U @ U.T, y)
        out[r] = exact_error(U.T @ beta, b, [mu1u, mu2u], var, theta)
    return float(out.mean()), float(out.std(ddof=1) / np.sqrt(n_rep))


def main():
    delta = 0.25
    n1 = n2 = 2
    n_rep = 200
    ds = [250, 1000, 4000, 16000, 64000]
    cs = [0.8, 0.5, 0.2, 1e-4]

    print("=== vs d (n1=n2=2, delta=0.25, 200 reps) ===")
    header = f"{'d':>8}" + "".join(
        f"{('c=' + str(c) if c >= 0.01 else 'non-spk'):>16}" for c in cs
    )
    print(header)
    print("-" * len(header))
    for i, d in enumerate(ds):
        row = f"{d:>8}"
        for j, c in enumerate(cs):
            mean, se = exp_error(d, c, delta, n1, n2, n_rep, seed=40 + 10 * i + j)
            row += f"{mean:8.4f} ({se:.4f})"
        print(row)

    print()
    print("=== vs n (d=4000, delta=0.25) ===")
    print(f"{'n':>4}{'c=0.8':>12}{'c=0.5':>12}{'non-spk':>12}")
    print("-" * 40)
    for k, n in enumerate([2, 3, 5, 10]):
        means = []
        for j, c in enumerate([0.8, 0.5, 1e-4]):
            mean, _ = exp_error(4000, c, delta, n, n, n_rep, seed=70 + 10 * k + j)
            means.append(mean)
        print(f"{n:>4}{means[0]:>12.4f}{means[1]:>12.4f}{means[2]:>12.4f}")


if __name__ == "__main__":
    main()
