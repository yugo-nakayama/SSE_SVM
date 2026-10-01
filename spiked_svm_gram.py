#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spiked_svm_gram.py

Section 11.2 — Convergence of the Gram matrix (Theorem 1).

Measures ||G - G*||_F for n1 = n2 = 2, m = 1, c = 0.5, delta = 1,
with spiked coordinates fixed within each replication.

Usage:
    python spiked_svm_gram.py
"""

import numpy as np

from spiked_svm_common import make_sd


def experiment_gram(n_rep=100, n1=2, n2=2, c=0.5, delta=1.0, seed=29):
    """Average ||G - G*||_F; under Theorem 1 the error is O(d^{-1/2})."""
    ds = [100, 1000, 10000, 100000, 1000000]
    N = n1 + n2
    eps = np.r_[np.ones(n1), -np.ones(n2)]
    out = {}
    for d in ds:
        rng = np.random.default_rng(seed)
        errs = np.empty(n_rep)
        for r in range(n_rep):
            z = rng.standard_normal(N)
            theta = float(d)
            sd, _, lams, _ = make_sd(d, [c])
            Delta = delta * theta
            mu1 = np.zeros(d)
            mu1[0] = np.sqrt(Delta) / 2.0
            X = np.empty((N, d))
            for p in range(N):
                x = rng.standard_normal(d) * sd
                x[1] = np.sqrt(lams[0]) * z[p]
                X[p] = (mu1 if eps[p] > 0 else -mu1) + x
            U = X / np.sqrt(theta)
            G = U @ U.T
            Gs = np.empty((N, N))
            for p in range(N):
                for q in range(N):
                    Gs[p, q] = eps[p] * eps[q] * delta / 4.0 + c * z[p] * z[q]
                Gs[p, p] += 1.0 - c
            errs[r] = np.linalg.norm(G - Gs, "fro")
        out[d] = (float(errs.mean()), float(errs.std(ddof=1) / np.sqrt(n_rep)))
    return ds, out


def main():
    ds, out = experiment_gram()
    print(f"{'d':>12}{'||G-G*||_F':>14}{'s.e.':>12}{'ratio':>10}")
    print("-" * 48)
    prev = None
    for d in ds:
        mean, se = out[d]
        ratio = (prev / mean) if prev is not None else float("nan")
        ratio_s = f"{ratio:10.3f}" if prev is not None else f"{'--':>10}"
        print(f"{d:>12}{mean:>14.5f}{se:>12.5f}{ratio_s}")
        prev = mean


if __name__ == "__main__":
    main()
