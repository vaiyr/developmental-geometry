#!/usr/bin/env bash
# Regenerate data/ from scratch: deploy → spawn all jobs → poll → analyze.
# Needs a configured Modal account with GPU access. ~60 min wall clock.
set -euo pipefail
cd "$(dirname "$0")"

modal deploy circle_geometry.py

python3 orchestrate.py spawn_geometry
python3 orchestrate.py spawn_llc_timeseries 1e-4 100 layers.7.

python3 orchestrate.py poll data/calls_geometry.json
python3 orchestrate.py poll data/calls_llc_timeseries.json

python3 analyze.py
