# CHESSFORMER-924K

Projet mini indépendant, consacré aux modèles de moins d'un million de
paramètres. Le candidat `geometric` contient **924 164 paramètres**. Le bot
Lichess utilise le compte **Chessformer-924K** et un passage neuronal par coup.

Le projet **CHESSFORMER-143M** reste dans [`../CHESSFORMER`](../CHESSFORMER).
Ce dossier possède son propre code, son environnement Python, ses données,
ses checkpoints, ses rapports et sa configuration Lichess.

## Installation et vérification

L'environnement `.venv` local est déjà préparé. Pour une nouvelle installation :

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,modal]'
```

Utiliser l'interpréteur de ce dossier, car les deux projets conservent le nom
de package Python `chessformer` pour la compatibilité des commandes et modèles.

```bash
.venv/bin/python -m pytest
.venv/bin/python -m chessformer.mini.train --help
.venv/bin/python -m chessformer.mini.compare --help
```

## Organisation

| Emplacement | Contenu |
|---|---|
| `src/chessformer/mini/` | Réseaux mini, données, entraînement, joueur, UCI et bot |
| `src/chessformer/*.py` | Copies locales des modules nécessaires : encodage, shards, pertes, matchs et transport Lichess |
| `train_mini_modal.py` | Entraînements mini sur Modal |
| `modal_support.py` | Image et fonctions Modal locales, sans import du lanceur 143M |
| `tools/run_mini*.sh` | Lancement depuis ce dossier, même avec un autre répertoire courant |
| `data/mini-*` | Corpus mini, y compris `mini-130m` |
| `checkpoints/mini-*` | Checkpoints des expériences mini |
| `docs/` | Protocole, mesures et rapports mini |
| `logs/` | Journaux mini |

Les modules issus du socle commun sont des fichiers indépendants, sans lien
vers le code 143M. Ils permettent notamment la distillation facultative d'un
professeur 143M hors partie et réutilisent le transport Lichess. Le joueur mini
reste défini dans `chessformer.mini.player`.

## Entraînement et évaluation

```bash
.venv/bin/python -m chessformer.mini.train \
  --candidate geometric --data data/mini-pilot-5m \
  --output checkpoints/mini-new-run --device cuda \
  --epochs 3 --batch-size 256 --lr 3e-4

.venv/bin/python -m chessformer.mini.ladder \
  --model geometric=checkpoints/mini-long-geometric-20260910/models/geometric/best.pt \
  --opponents material1 stockfish1320 --stockfish tools/stockfish \
  --output docs/mini-new-ladder.json
```

Le [protocole et l'historique des expériences](docs/MINI_PILOT.md) détaillent
les candidats, la préparation des corpus et les résultats.

Les corpus préparés sont disponibles dans ce dossier. Pour régénérer un corpus
depuis les annotations Stockfish d'origine, sélectionner explicitement la source :

```bash
.venv/bin/python -m chessformer.mini.data \
  --source /home/ubuntu/CHESSFORMER/data/eval-d3-multipv \
  --output data/mini-new-data
```

Cette importation facultative de données n'est nécessaire ni à l'utilisation
des corpus existants ni au jeu. Les chemins de provenance contenus dans les
manifestes et les rapports historiques restent ceux enregistrés lors des runs ;
ils sont conservés pour préserver les empreintes des datasets et les reprises.

## Compte Lichess

`.env` contient uniquement `LICHESS_TOKEN_MINI` et les identifiants Modal.
Le token du compte 143M reste dans l'autre projet. Le bot et le script de
challenges vérifient le compte Chessformer-924K avant toute partie.

```bash
tools/run_mini_bot.sh
tools/run_mini_ladder.sh --rounds 1 --min-rating 1400 --max-rating 1900
```

Le bot doit être lancé une seule fois. Le 2 octobre 2026, l’ancien processus
a été arrêté et le bot a été relancé depuis ce dossier avec son propre `.venv`,
le même checkpoint et les mêmes paramètres de jeu. Un tour de ladder blitz
3+2 a été lancé et les coups ont été vérifiés dans le journal local.
Le bot a ensuite été arrêté à la demande de l'utilisateur à 18 h 46 (Paris).
Les futurs lancements utilisent `tools/run_mini_bot.sh` de ce projet.

## Modal

L'application reste `chessformer-mini`. Les jeux de données et runs distants
existants restent dans le volume historique `chessformer-v1`, sous les noms
`mini-*`. La séparation locale ne migre pas ce volume et ne lance aucun calcul.
Utiliser des noms `mini-*` pour les futurs datasets et runs de ce projet.

Le [plan pour viser 2000 Lichess blitz en 48 heures](docs/PLAN_2000_ELO_48H_2026-10-02.md)
documente les résultats, les limites et l'essai B300 autorisé sous 40 $.
`train_mini_b300_modal.py` conserve une échéance cumulée après préemption ;
`tools/monitor_mini_b300.py` arrête ce run à son échéance et évalue ses poids
localement, en gardant le bot hors ligne.

## Séparation du 2 octobre 2026

Le [journal de séparation](docs/PROJECT_SEPARATION.md) décrit le périmètre,
les sauvegardes et la validation. Les données et les checkpoints ont été
déplacés sur le même disque, avec conservation des fichiers et des liens durs.
