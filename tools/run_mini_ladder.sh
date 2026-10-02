#!/bin/bash
# Rated challenges for your mini BOT; keep tools/run_mini_bot.sh online.
#
#   tools/run_mini_ladder.sh --rounds 2 --min-rating 1400 --max-rating 1900
#
# Same .env and account validation as the bot. CLI flags override these defaults.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

exec .venv/bin/python -u tools/challenge_ladder.py \
  --rounds 1 --size 3 --min-rating 1400 --max-rating 2100 \
  --clock-limit 180 --clock-increment 2 --perf blitz \
  --settle-s 5 --accept-timeout-s 30 --duel "$@"
