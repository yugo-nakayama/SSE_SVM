#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spiked_svm_common.py

Shared utilities for the numerical experiments in Section 11.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def solve_hard_margin(G, y):
    """Solve the dual problem (5.2) and return (beta, b, alpha).

    max 1'a - a'YGYa/2  subject to  a >= 0,  y'a = 0,
    and beta = a * y, b = mean over the support vectors of y - G beta.
    """
    N = len(y)
    Q = (y[:, None] * G) * y[None, :]
    fun = lambda a: -(a.sum() - 0.5 * a @ Q @ a)
    jac = lambda a: -(np.ones(N) - Q @ a)
    cons = [{"type": "eq", "fun": lambda a: y @ a, "jac": lambda a: y}]
    res = minimize(
        fun,
        np.full(N, 0.1),
        jac=jac,
        bounds=[(0, None)] * N,
        constraints=cons,
        method="SLSQP",
        options={"maxiter": 800, "ftol": 1e-12},
    )
    a = np.maximum(res.x, 0.0)
    beta = a * y
    sv = a > 1e-7 * max(a.max(), 1e-12)
    if sv.sum() == 0:
        sv = a >= a.max()
    b = float(np.mean(y[sv] - (G @ beta)[sv]))
    return beta, b, a


def exact_error(w, b, mu_list, var, theta):
    """Exact e(1)+e(2) for f(u) = w'u + b with u = x/sqrt(theta)."""
    e = 0.0
    v = float(np.sum(var * w ** 2) / theta)
    for mu0, eps in zip(mu_list, (1.0, -1.0)):
        e += norm.cdf(-eps * float(w @ mu0 + b) / np.sqrt(v))
    return e


def make_sd(d, c_list, gamma=1.0, theta=None):
    """Sigma = diag(sig2, lam_1,..,lam_m, sig2,..) as in (11.1)."""
    theta = float(d) if theta is None else float(theta)
    m = len(c_list)
    lams = np.array([c * theta * gamma for c in c_list], dtype=float)
    sig2 = (gamma * theta - lams.sum()) / (d - m)
    var = np.full(d, sig2)
    var[1 : 1 + m] = lams
    return np.sqrt(var), var, lams, sig2


def tr_of(X):
    """Trace of the sample covariance matrix of X."""
    n = X.shape[0]
    Xc = X - X.mean(axis=0)
    return float((Xc * Xc).sum() / (n - 1))


def nr_estimate(X, m):
    """Noise-reduction estimation of spiked eigenvalues and eigenvectors.

    X : n x d data matrix of one class. Returns (lam_tilde, H_hat, tr_S).
    """
    n = X.shape[0]
    Xc = X - X.mean(axis=0)
    SD = (Xc @ Xc.T) / (n - 1)
    ev, U = np.linalg.eigh(SD)
    idx = np.argsort(ev)[::-1]
    ev, U = ev[idx], U[:, idx]
    trS = float(ev.sum())
    lam_t = np.empty(m)
    for s in range(m):
        rest = trS - ev[: s + 1].sum()
        den = n - 1 - (s + 1)
        lam_t[s] = ev[s] - (rest / den if den > 0 else 0.0)
    lam_t = np.maximum(lam_t, 1e-12)
    H = np.empty((X.shape[1], m))
    for s in range(m):
        v = Xc.T @ U[:, s]
        H[:, s] = v / max(np.linalg.norm(v), 1e-300)
    return lam_t, H, trS
