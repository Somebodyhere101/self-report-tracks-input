#!/bin/bash
# Operational driver (no analysis logic). Runs the chain pilot -> main -> analysis. Each GPU job goes through
# the shared lock (labels p9-pilot-<k>, p9-main-<k>); a job that reaches its time budget exits 3 and is
# re-queued, so every job stays under about 60 minutes and other sessions can interleave. Before each job it
# waits for mains power if the laptop is discharging below 30%. It stops if ../STOP exists, a job fails, or
# a frozen bed check fails (make_config.py). Usage: bash run_chunks.sh
cd "$(dirname "$0")"
PY=/c/Users/caabm/.venvs/jspace/Scripts/python.exe
LOCK=../../../tools/gpu_lock.py
LOG=../logs/run_chunks.log
bat() { powershell -NoProfile -Command '$b=Get-CimInstance Win32_Battery; "$($b.BatteryStatus) $($b.EstimatedChargeRemaining)"' | tr -d '\r'; }
wait_power() {
  while true; do
    read -r st ch <<< "$(bat)"
    if [ "$st" = "1" ] && [ "${ch:-100}" -lt 30 ]; then
      echo "$(date '+%F %T') waiting for mains power (battery ${ch}%)" >> "$LOG"; sleep 120
    else
      return
    fi
  done
}
run_phase() {  # $1 = pilot | main
  local k=1
  while true; do
    [ -f ../STOP ] && { echo "$(date '+%F %T') STOP file present; stopping before $1-$k" >> "$LOG"; exit 0; }
    wait_power
    echo "$(date '+%F %T') queue $1-$k" >> "$LOG"
    PYTHONIOENCODING=utf-8 $PY $LOCK --label "p9-$1-$k" -- $PY job.py "$1" >> "../logs/job_$1.log" 2>&1
    rc=$?
    echo "$(date '+%F %T') end $1-$k rc=$rc" >> "$LOG"
    [ $rc -eq 0 ] && return 0
    [ $rc -eq 3 ] && { k=$((k + 1)); sleep 20; continue; }
    exit $rc
  done
}
[ -f ../res_pilot/PILOT_DONE ] || run_phase pilot
[ -f ../res/MAIN_DONE ] || run_phase main
echo "$(date '+%F %T') analysis" >> "$LOG"
PYTHONIOENCODING=utf-8 $PY analyze.py > ../logs/analyze.log 2>&1
echo "$(date '+%F %T') analysis rc=$? ; chain complete" >> "$LOG"
