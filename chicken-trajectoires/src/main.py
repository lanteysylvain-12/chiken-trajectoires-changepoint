import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulate import simulate, features
from cpd import binseg

os.makedirs("results", exist_ok=True)

# --- 1. Une trajectoire illustrative ---
xy, reg = simulate(n=300, taus=(150,), seed=3)
cps = binseg(features(xy))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].plot(*xy.T, color="lightgray", lw=0.8)
ax[0].scatter(*xy[:-1].T, c=reg, cmap="coolwarm", s=8); ax[0].set_aspect("equal"); ax[0].set_title("Trajectoire (bleu = actif, rouge = repos)")
f = features(xy)
ax[1].plot(f[:, 0], lw=0.8, label="log(longueur de pas)")
for c in cps: ax[1].axvline(c, color="k", ls="--", label="rupture estimée" if c == cps[0] else None)
ax[1].axvline(150, color="g", label="rupture réelle"); ax[1].legend(); plt.tight_layout(); plt.savefig("results/trajectoire.png", dpi=120)

# --- 2. Monte-Carlo : précision de localisation et fausses détections ---
errs, nb = [], []
for s in range(200):
    xy, _ = simulate(n=300, taus=(150,), seed=100 + s)
    c = binseg(features(xy)); nb.append(len(c))
    if c: errs.append(min(c, key=lambda t: abs(t - 150)) - 150)
errs = np.array(errs)
H0 = [len(binseg(features(simulate(n=300, taus=(), seed=500 + s)[0]))) for s in range(200)]
out = {"erreur_absolue_mediane": float(np.median(np.abs(errs))), "erreur_absolue_moyenne": float(np.mean(np.abs(errs))),
       "taux_detection_vraie_rupture": float(np.mean(np.array(nb) >= 1)), "taux_fausses_alarmes_sans_rupture": float(np.mean(np.array(H0) >= 1))}
print(json.dumps(out, indent=2)); json.dump(out, open("results/metrics.json", "w"), indent=2)
plt.figure(figsize=(6, 3.5)); plt.hist(errs, bins=25); plt.xlabel("t̂ - τ"); plt.title("Erreur de localisation (200 simulations)")
plt.tight_layout(); plt.savefig("results/erreurs.png", dpi=120)
