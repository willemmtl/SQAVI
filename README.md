# SQAVI
Serial Quadratic Approximations for Variational Inference

## Démarrage

**IMPORTANT** : Les résultats mcmc (`mcmc_chain.dat` dans les dossiers `results/200x200` et `results/casr`) n'ont **pas pu être stockés** sur le repo distant car ils sont **trop volumineux**. Il faut donc soit
- les recréer comme indiqué dans le calepin `memoire.ipynb` (peut prendre beaucoup de temps),
- les ajouter manuellement si on les a.

Les fichiers `empty.bin` des dossiers `results/200x200` et `results/casr` peuvent être supprimés. Ils ne sont là que pour pull le repo avec la bonne structure de dossiers.

Par défaut, de nombreux dossiers et fichiers qui pourraient être volumineux sont ignorés par git afin d'éviter des transferts trop volumineux. Il est laissé à la discrétion de l'utilisateur de modifier le fichier `.gitignore` pour chnager cela.