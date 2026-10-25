#!/bin/bash
# Code dry run on CPU with SmolLM2-135M (P9_TINY=1): every stage once, outputs in *_tiny folders. No GPU.
cd "$(dirname "$0")"
PY=/c/Users/caabm/.venvs/jspace/Scripts/python.exe
export P9_TINY=1 PYTHONIOENCODING=utf-8
set -e
$PY stage_screen.py
$PY stage_pilot.py
$PY stage_gen.py --selftest
for A in O I J X; do $PY stage_train.py --arm $A --seed 0 --selftest; done
for A in O I J X; do $PY stage_eval.py --arm $A --seed 0 --selftest; done
$PY stage_eval.py --arm N --selftest
$PY analyze.py --selftest
echo "DRYRUN DONE"
