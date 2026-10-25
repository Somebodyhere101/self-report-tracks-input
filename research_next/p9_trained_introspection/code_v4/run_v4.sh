#!/bin/bash
# Operational driver for v4 (no analysis logic): pilot (budget ladder + learnability gate) -> main -> CPU analysis.
# Each GPU job goes through the shared lock with an explicit time limit (labels p9v4-pilot-k, p9v4-main-k). Exit 3
# (deadline near, state saved) or 124 (killed by the lock after an overrun; checkpoints survive) means re-queue.
# Exit 2 from the pilot means the learnability gate was not passed: the main phase is skipped and the analysis
# reports UNINFORMATIVE. Waits for mains power if discharging below 30%; stops if ../STOP exists or a job fails.
cd "$(dirname "$0")"
PY=/c/Users/<user>/.venvs/jspace/Scripts/python.exe
LOCK=../../../tools/gpu_lock.py
LOG=../logs/run_v4.log
MAXMIN=90
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
    $PY $LOCK --label "p9v4-$1-$k" --max-min $MAXMIN -- $PY job_v4.py "$1" >> "../logs/job_v4_$1.log" 2>&1
    rc=$?
    echo "$(date '+%F %T') end $1-$k rc=$rc" >> "$LOG"
    [ $rc -eq 0 ] && return 0
    { [ $rc -eq 3 ] || [ $rc -eq 124 ]; } && { k=$((k + 1)); sleep 20; continue; }
    return $rc
  done
}
mkdir -p ../logs
if [ ! -f ../res_pilot_v4/PILOT_DONE ]; then
  run_phase pilot
  rc=$?
  [ $rc -eq 0 ] || [ $rc -eq 2 ] || exit $rc
fi
if [ -f ../res_v4/NO_RUNG ]; then
  echo "$(date '+%F %T') learnability gate not passed: main phase skipped" >> "$LOG"
else
  [ -f ../res_v4/MAIN_DONE ] || run_phase main || exit $?
fi
$PY v4_analyze.py > ../logs/analyze_v4.log 2>&1
echo "$(date '+%F %T') analysis rc=$? ; chain complete" >> "$LOG"
