#!/bin/bash
# Operational driver for v2 (no analysis logic): wait for mains power if discharging below 30%, run job_v2.py
# through the shared GPU lock (label p9v2-premise), then the CPU analysis. Stops if ../STOP exists.
cd "$(dirname "$0")"
PY=/c/Users/caabm/.venvs/jspace/Scripts/python.exe
LOCK=../../../tools/gpu_lock.py
LOG=../logs/run_v2.log
bat() { powershell -NoProfile -Command '$b=Get-CimInstance Win32_Battery; "$($b.BatteryStatus) $($b.EstimatedChargeRemaining)"' | tr -d '\r'; }
[ -f ../STOP ] && { echo "$(date '+%F %T') STOP present" >> "$LOG"; exit 0; }
while true; do
  read -r st ch <<< "$(bat)"
  if [ "$st" = "1" ] && [ "${ch:-100}" -lt 30 ]; then echo "$(date '+%F %T') waiting for mains" >> "$LOG"; sleep 120; else break; fi
done
echo "$(date '+%F %T') queue p9v2-premise" >> "$LOG"
PYTHONIOENCODING=utf-8 $PY $LOCK --label p9v2-premise -- $PY job_v2.py >> ../logs/job_v2.log 2>&1
rc=$?
echo "$(date '+%F %T') end p9v2-premise rc=$rc" >> "$LOG"
[ $rc -ne 0 ] && exit $rc
PYTHONIOENCODING=utf-8 $PY analyze_v2.py > ../logs/analyze_v2.log 2>&1
echo "$(date '+%F %T') analysis rc=$?" >> "$LOG"
