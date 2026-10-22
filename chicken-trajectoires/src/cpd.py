"""Détection de ruptures : rapport de vraisemblance gaussien (moyenne + variance) + segmentation binaire pénalisée."""
import numpy as np


def _cost_factory(Z):
    c1 = np.vstack([np.zeros(Z.shape[1]), np.cumsum(Z, 0)])
    c2 = np.vstack([np.zeros(Z.shape[1]), np.cumsum(Z ** 2, 0)])

    def cost(a, b):  # -2 loglik (à une constante près) du segment [a, b)
        n = b - a
        m = (c1[b] - c1[a]) / n
        v = np.maximum((c2[b] - c2[a]) / n - m ** 2, 1e-8)
        return n * np.log(v).sum()
    return cost


def best_split(cost, a, b, min_size):
    best, arg = np.inf, None
    for t in range(a + min_size, b - min_size + 1):
        c = cost(a, t) + cost(t, b)
        if c < best:
            best, arg = c, t
    return arg, cost(a, b) - best  # gain


def binseg(Z, min_size=15, pen_factor=3.0):
    n, d = Z.shape
    pen = pen_factor * (2 * d + 1) * np.log(n)
    cost = _cost_factory(Z)
    cps, stack = [], [(0, n)]
    while stack:
        a, b = stack.pop()
        if b - a < 2 * min_size:
            continue
        t, gain = best_split(cost, a, b, min_size)
        if t is not None and gain > pen:
            cps.append(t); stack += [(a, t), (t, b)]
    return sorted(cps)
