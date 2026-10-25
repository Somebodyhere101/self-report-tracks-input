#!/bin/bash
# Code dry run of the post-pilot stages only (P9_TINY=1, CPU, SmolLM2-135M), reusing res_tiny and res_pilot_tiny.
cd "$(dirname "$0")"
PY=/c/Users/caabm/.venvs/jspace/Scripts/python.exe
export P9_TINY=1 PYTHONIOENCODING=utf-8
set -e
$PY stage_gen.py --selftest
for A in O I J X; do $PY stage_train.py --arm $A --seed 0 --selftest; done
for A in O I J X; do $PY stage_eval.py --arm $A --seed 0 --selftest; done
$PY stage_eval.py --arm N --selftest
$PY analyze.py --selftest
$PY report_tables.py --selftest > /dev/null
echo "RERUN DONE"
