# Modélisation stochastique de trajectoires et détection de ruptures comportementales

Marche aléatoire corrélée (longueurs de pas Gamma, angles de virage de von Mises) avec changement de régime « actif » → « repos ». Détection de ruptures par rapport de vraisemblance gaussien sur (log longueur de pas, cos de l'angle de virage), segmentation binaire et pénalité de type BIC.

## Lancer
```bash
pip install -r requirements.txt
python src/main.py
```

## Résultats (200 simulations, n = 300, rupture à t = 150)
- Rupture détectée dans 84 % des cas ; erreur absolue médiane de localisation : 4 pas
- 0 fausse alarme sur 200 trajectoires sans rupture

## Pistes
Modèles à états cachés (HMM / HSMM), plusieurs ruptures, estimation des paramètres par maximum de vraisemblance, données réelles de suivi d'animaux.
