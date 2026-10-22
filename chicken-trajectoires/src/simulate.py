"""Marche aléatoire corrélée avec changement de régime (actif -> repos), type trajectoire d'animal."""
import numpy as np

REGIMES = {"actif": dict(step=1.0, kappa=3.0), "repos": dict(step=0.7, kappa=1.5)}


def simulate(n=300, taus=(150,), seed=0, regimes=("actif", "repos")):
    rng = np.random.default_rng(seed)
    bounds = [0, *taus, n]
    reg = np.zeros(n, dtype=int)
    for k in range(len(bounds) - 1):
        reg[bounds[k]:bounds[k + 1]] = k % len(regimes)
    xy = np.zeros((n + 1, 2)); heading = rng.uniform(0, 2 * np.pi)
    for t in range(n):
        p = REGIMES[regimes[reg[t]]]
        heading += rng.vonmises(0, p["kappa"])
        s = rng.gamma(4.0, p["step"] / 4.0)
        xy[t + 1] = xy[t] + s * np.array([np.cos(heading), np.sin(heading)])
    return xy, reg


def features(xy):
    d = np.diff(xy, axis=0)
    step = np.linalg.norm(d, axis=1)
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.diff(ang, prepend=ang[0])
    return np.column_stack([np.log(step + 1e-9), np.cos(turn)])
