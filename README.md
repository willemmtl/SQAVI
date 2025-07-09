# SQAVI
Serial Quadratic Approximations for Variational Inference

## Démarrage

### IMPORTANT
Les résultats mcmc (`mcmc_chain.dat` dans les dossiers `results/200x200` et `results/casr`) n'ont **pas pu être stockés** sur le repo distant car ils sont **trop volumineux**. Il faut donc soit
- les recréer comme indiqué dans le calepin `memoire.ipynb` (peut prendre beaucoup de temps),
- les ajouter manuellement si on les a.

### Structure du repo
- Le fichier `memoire.ipynb` est le calepin principal qu'il faut exécuter pour (re)générer les résultats et principaux tracés du mémoire. Il est écrit en **julia**.
- Le fichier `visualisation.ipynb` est un calepin permettant de (re)générer les cartes de chaleur **associées aux données CaSR** du mémoire. Il est rédigé en **python**.
- Le dossier `src` contient les fonctions **fondamentales qui ne dépendent pas du jeu de données considérées**. Il contient notamment le code source pour les algorithmes SQAVI et MCMC utilisés.
- Le dossier `casr` contient les fonctions **spécifiques au jeu de données CaSR**, aussi bien pour le preprocessing que pour afficher les résultats (dossier `casr/src`). Il contient également les **données** pré-traitées (dossier `casr/data/preprocessed`) et certaines données utiles à la visualisation dans l'espace (dossier `casr/data/viz`).

### Autres détails
Les fichiers `empty.bin` des dossiers `results/200x200`, `results/casr` et `heatmaps` peuvent être supprimés. Ils sont là uniquement pour donner une bonne structure de dossiers.

Par défaut, de nombreux dossiers et fichiers qui pourraient être volumineux sont ignorés par git afin d'éviter des transferts trop volumineux. Il est laissé à la discrétion de l'utilisateur de modifier le fichier `.gitignore` pour chnager cela.

## Données

Les données brutes de CaSR v3.1 ont été extraites via [ce site](https://hpfx.collab.science.gc.ca/~scar700/rcas-casr/download_CaSRv3.1_regions_var_period.html) en sélectionnant une zone rectangulaire recouvrant le Québec. Un dossier compressé contenant les données devrait être dans les main de Jonathan si besoin :).