#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spiked_svm_classifiers.py

Section 11.5 — Comparison of classifiers.

Compares SVM, BC-SVM(kappa), BC-SVM(kappa_*), SC-SVM and SC-SVM(split).

(A) Fixed (n1, n2) = (12, 6), vary c (Table compare).
(B) Fixed c = 0.8, delta = 0.01, vary (n1, n2) (Figure n-scale / dropped Table 8).

Usage:
    python spiked_svm_classifiers.py
"""

import numpy as np

from spiked_svm_common import (
    exact_error,
    make_sd,
    nr_estimate,
    solve_hard_margin,
    tr_of,
)


def exp_classifiers(d, c, delta, n1, n2, n_rep, seed, m=1):
    """Compare five procedures; returns means and s.e."""
    rng = np.random.default_rng(seed)
    theta = float(d)
    sd, var, _, _ = make_sd(d, [c] if c > 0 else [1e-12])
    Delta = delta * theta
    mu1 = np.zeros(d)
    mu1[0] = np.sqrt(Delta) / 2.0
    mu1u, mu2u = mu1 / np.sqrt(theta), -mu1 / np.sqrt(theta)
    y = np.r_[np.ones(n1), -np.ones(n2)]
    keys = ["SVM", "BC-SVM(kappa)", "BC-SVM(kappa*)", "SC-SVM", "SC-SVM(split)"]
    acc = {k: np.empty(n_rep) for k in keys}

    for r in range(n_rep):
        X1 = mu1 + rng.standard_normal((n1, d)) * sd
        X2 = -mu1 + rng.standard_normal((n2, d)) * sd

        U = np.vstack([X1, X2]) / np.sqrt(theta)
        beta, b, a = solve_hard_margin(U @ U.T, y)
        w = U.T @ beta
        acc["SVM"][r] = exact_error(w, b, [mu1u, mu2u], var, theta)

        abar1, abar2 = a[:n1].mean(), a[n1:].mean()
        lt1, _, trS1 = nr_estimate(X1, m)
        lt2, _, trS2 = nr_estimate(X2, m)
        corr_f = 0.5 * (abar1 * trS1 / theta - abar2 * trS2 / theta)
        acc["BC-SVM(kappa)"][r] = exact_error(
            w, b + corr_f, [mu1u, mu2u], var, theta
        )
        c10s = (trS1 - lt1.sum()) / theta
        c20s = (trS2 - lt2.sum()) / theta
        corr_s = 0.5 * (abar1 * c10s - abar2 * c20s)
        acc["BC-SVM(kappa*)"][r] = exact_error(
            w, b + corr_s, [mu1u, mu2u], var, theta
        )

        def sc_run(E1, E2, T1, T2):
            lt1e, H1, _ = nr_estimate(E1, m)
            lt2e, H2, _ = nr_estimate(E2, m)
            muhat = E1.mean(axis=0) - E2.mean(axis=0)
            cols = []
            for lt, H in ((lt1e, H1), (lt2e, H2)):
                for s_ in range(m):
                    if (H[:, s_] @ muhat) ** 2 / lt[s_] <= np.log(d):
                        cols.append(H[:, s_])
            Um = (
                np.linalg.qr(np.array(cols).T)[0]
                if cols
                else np.zeros((d, 0))
            )
            pr = (
                (lambda Z: Z - (Z @ Um) @ Um.T)
                if Um.shape[1]
                else (lambda Z: Z)
            )
            P1, P2 = pr(T1), pr(T2)
            yy = np.r_[np.ones(len(P1)), -np.ones(len(P2))]
            V = np.vstack([P1, P2]) / np.sqrt(theta)
            bt, bb, aa = solve_hard_margin(V @ V.T, yy)
            wv = V.T @ bt
            k1 = len(P1)
            corr = 0.5 * (
                aa[:k1].mean() * tr_of(P1) / theta
                - aa[k1:].mean() * tr_of(P2) / theta
            )
            return exact_error(
                wv,
                bb + corr,
                [pr(mu1u[None, :])[0], pr(mu2u[None, :])[0]],
                var,
                theta,
            )

        acc["SC-SVM"][r] = sc_run(X1, X2, X1, X2)
        h1, h2 = n1 // 2, n2 // 2
        acc["SC-SVM(split)"][r] = sc_run(X1[:h1], X2[:h2], X1[h1:], X2[h2:])

    return {
        k: (float(v.mean()), float(v.std(ddof=1) / np.sqrt(n_rep)))
        for k, v in acc.items()
    }


def _print_row(label, res):
    keys = ["SVM", "BC-SVM(kappa)", "BC-SVM(kappa*)", "SC-SVM", "SC-SVM(split)"]
    parts = [f"{label:<12}"]
    for k in keys:
        mean, se = res[k]
        parts.append(f"{mean:7.4f}({se:.4f})")
    print(" ".join(parts))


def main():
    d = 20000
    n_rep = 100

    print("=== Table compare: n1=12, n2=6, delta=0.05 ===")
    keys = ["SVM", "BC(k)", "BC(k*)", "SC", "SC(split)"]
    print(f"{'':12}" + "".join(f"{k:>14}" for k in keys))
    for i, c in enumerate([0.0, 0.2, 0.5, 0.8]):
        label = "non-spiked" if c == 0.0 else f"c={c}"
        res = exp_classifiers(
            d, c, 0.05, 12, 6, n_rep, seed=100 + i, m=1
        )
        _print_row(label, res)

    print()
    print("=== Figure n-scale: c=0.8, delta=0.01 ===")
    print(f"{'(n1,n2)':12}" + "".join(f"{k:>14}" for k in keys))
    for i, (n1, n2) in enumerate([(12, 6), (24, 12), (48, 24)]):
        res = exp_classifiers(
            d, 0.8, 0.01, n1, n2, n_rep, seed=200 + i, m=1
        )
        _print_row(f"({n1},{n2})", res)


if __name__ == "__main__":
    main()
