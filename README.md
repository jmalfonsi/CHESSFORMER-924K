# CHESSFORMER-924K

Bot d’échecs de **924 164 paramètres** : un passage neuronal par coup, masque des
coups légaux, puis choix du meilleur score. Cette version fournit les poids
actuellement utilisés par [Chessformer-924K](https://lichess.org/@/Chessformer-924K),
pour jouer avec votre propre compte BOT.

## Installation

Linux, ou macOS 14+ sur Apple Silicon, avec Bash et Python **3.12 ou plus**.
Le bot joue sur CPU.
Les versions des dépendances sont fixées à celles de la référence actuelle.

```bash
git clone https://github.com/jmalfonsi/CHESSFORMER-924K.git
cd CHESSFORMER-924K
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e .
cp .env.example .env
```

## Votre compte Lichess

Utilisez un compte **BOT dédié**. Pour en créer un, le compte ne doit avoir joué
aucune partie ; sa [conversion en BOT](https://lichess.org/api#tag/Bot/operation/botAccountUpgrade)
est irréversible.

Connecté à ce compte, créez un [token personnel](https://lichess.org/account/oauth/token/create)
avec les permissions **bot:play** (« Play bot moves ») et **challenge:write**
(« Create, accept, decline challenges »). Renseignez `.env` :

```dotenv
LICHESS_ACCOUNT_MINI=VotreNomDeBot
LICHESS_TOKEN_MINI=votre_token_personnel
```

Le bot et le ladder utilisent cette même configuration et vérifient que le token
appartient au nom indiqué et que le compte porte le titre BOT. Les variables
exportées dans le shell prennent priorité sur `.env`. Le fichier `.env` est
ignoré par Git ; gardez votre token privé.

## Jouer et lancer le ladder

Dans un premier terminal, laissez le bot connecté :

```bash
tools/run_mini_bot.sh
```

Dans un second terminal, depuis le même dossier :

```bash
tools/run_mini_ladder.sh
```

Le ladder propose **une ronde de trois adversaires**, choisis parmi les bots
en ligne entre **1400 et 2100 Elo blitz**, avec au moins 300 parties classées
et un classement établi. Les parties sont classées, en **3 minutes + 2 secondes**,
et se suivent une par une. Les adversaires peuvent refuser ; le ladder nécessite
que votre bot reste connecté dans le premier terminal.

Pour prolonger la session ou changer les adversaires :

```bash
tools/run_mini_ladder.sh --rounds 10 --size 10 --min-rating 1500 --max-rating 2000
```

`Ctrl+C` arrête le programme du terminal concerné. Arrêter le ladder laisse le
bot connecté ; attendez la fin des parties avant d’arrêter le bot. Les journaux
du bot sont enregistrés dans `logs/`.

## Modèle fourni

`models/chessformer-924k-v1.pt` contient les poids d’inférence actuels, sans état
d’optimiseur. Son empreinte et sa configuration figurent dans
[models/manifest.json](models/manifest.json).

Les réglages par défaut conservent le jeu actuel : température **0**, un thread
CPU, au plus cinq parties simultanées ; un coup causant une troisième répétition
est écarté lorsque la tête de valeur évalue la position au-dessus de zéro.
Le nom du compte ne change ni les poids ni les décisions du réseau. Le classement
de votre compte se construit avec ses propres parties.

Un autre checkpoint compatible peut être choisi explicitement :

```bash
tools/run_mini_bot.sh chemin/vers/best.pt
```

## Vérification locale

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
```
