# SQAVI
Serial Quadratic Approximations for Variational Inference

## Démarrage

### IMPORTANT
Les résultats mcmc (`mcmc_chain.dat` dans les dossiers `results/200x200` et `results/casr`) n'ont **pas pu être stockés** sur le repo distant car ils sont **trop volumineux**. Il faut donc soit
- les recréer comme indiqué dans le calepin `memoire.ipynb` (peut prendre beaucoup de temps),
- les ajouter manuellement si on les a.

### Structure du repo
- Le dossier `src` contient les fonctions **fondamentales qui ne dépendent pas du jeu de données considérées**. Il contient notamment le code source pour les algorithmes SQAVI et MCMC utilisés.
- Le dossier `casr` contient les fonctions **spécifiques au jeu de données CaSR**, aussi bien pour le preprocessing que pour afficher les résultats (dossier `casr/src`). Il contient également les **données** pré-traitées (dossier `casr/data/preprocessed`) et certaines données utiles à la visualisation dans l'espace (dossier `casr/data/viz`).

### Autres détails
Les fichiers `empty.bin` des dossiers `results/200x200` et `results/casr` peuvent être supprimés. Ils ne sont là que pour pull le repo avec la bonne structure de dossiers.

Par défaut, de nombreux dossiers et fichiers qui pourraient être volumineux sont ignorés par git afin d'éviter des transferts trop volumineux. Il est laissé à la discrétion de l'utilisateur de modifier le fichier `.gitignore` pour chnager cela.