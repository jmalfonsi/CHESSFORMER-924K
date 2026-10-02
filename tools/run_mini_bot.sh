#!/bin/bash
# Put your mini BOT online with the released weights and current playing settings.
#
#   tools/run_mini_bot.sh
#   tools/run_mini_bot.sh path/to/best.pt
#
# The account and token come from LICHESS_ACCOUNT_MINI / LICHESS_TOKEN_MINI
# in .env. The Python side checks the token's owner before playing.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

CHECKPOINT="${1:-models/chessformer-924k-v1.pt}"
# Temperature zero preserves the currently deployed deterministic policy.
TEMPERATURE="${TEMPERATURE:-0}"
SAMPLE_PLIES="${SAMPLE_PLIES:-20}"
# Withhold a third repetition when the value head rates the position above 0.
AVOID_REPETITION="${AVOID_REPETITION:-0}"
MAX_GAMES="${MAX_GAMES:-5}"

STAMP="$(date -u +%Y%m%d-%H%M%S)"
mkdir -p logs
LOG="logs/mini-bot-$STAMP.log"
echo "temperature=$TEMPERATURE sample_plies=$SAMPLE_PLIES avoid_repetition=$AVOID_REPETITION games=$MAX_GAMES checkpoint=$CHECKPOINT log=$LOG"
exec .venv/bin/python -m chessformer.mini.lichess --checkpoint "$CHECKPOINT" \
  --temperature "$TEMPERATURE" --sample-plies "$SAMPLE_PLIES" --avoid-repetition "$AVOID_REPETITION" \
  --max-games "$MAX_GAMES" 2>&1 | tee "$LOG"
