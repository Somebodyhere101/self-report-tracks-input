#!/bin/bash
# Operational driver for v3 (no analysis logic): pilot phase -> main phase -> CPU analysis. Each GPU job goes
# through the shared lock (labels p9v3-pilot-k, p9v3-main-k); exit 3 means "time budget reached, re-queue".
# Waits for mains power if discharging below 30%; stops if ../STOP exists, a job fails, or a bed check fails.
cd "$(dirname "$0")"
PY=/c/Users/caabm/.venvs/jspace/Scripts/python.exe
LOCK=../../../tools/gpu_lock.py
LOG=../logs/run_v3.log
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONIOENCODING=utf-8
bat() { powershell -NoProfile -Command '$b=Get-CimInstance Win32_Battery; "$($b.BatteryStatus) $($b.EstimatedChargeRemaining)"' | tr -d '\r'; }
wait_power() {
  while true; do
    read -r st ch <<< "$(bat)"
    if [ "$st" = "1" ] && [ "${ch:-100}" -lt 30 ]; then echo "$(date '+%F %T') waiting for mains" >> "$LOG"; sleep 120; else return; fi
  done
}
run_phase() {
  local k=1
  while true; do
    [ -f ../STOP ] && { echo "$(date '+%F %T') STOP present before $1-$k" >> "$LOG"; exit 0; }
    wait_power
    echo "$(date '+%F %T') queue $1-$k" >> "$LOG"
    $PY $LOCK --label "p9v3-$1-$k" -- $PY job_v3.py "$1" >> "../logs/job_v3_$1.log" 2>&1
    rc=$?
    echo "$(date '+%F %T') end $1-$k rc=$rc" >> "$LOG"
    [ $rc -eq 0 ] && return 0
    [ $rc -eq 3 ] && { k=$((k + 1)); sleep 20; continue; }
    exit $rc
  done
}
[ -f ../res_pilot_v3/PILOT_DONE ] || run_phase pilot
[ -f ../res_v3/MAIN_DONE ] || run_phase main
$PY v3_analyze.py > ../logs/analyze_v3.log 2>&1
echo "$(date '+%F %T') analysis rc=$? ; chain complete" >> "$LOG"
