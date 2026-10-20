#!/usr/bin/env bash
# Usage: run_guarded.sh <label> <python-script> [args...]
# Waits (<=30 min) for >=1.5 GiB free Windows RAM, then runs the script under the shared heavyjob lock
# using the copy's code first and the original venv. Never edits the original project.
set -u
COPY=<PROJECT_ROOT>
ORIG=<CANONICAL_ROOT>
label="$1"; shift
script="$1"; shift
winfree() { powershell.exe -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)" 2>/dev/null | tr -d '\r'; }
for _ in $(seq 1 60); do
  f=$(winfree); f=${f:-0}
  if awk "BEGIN{exit !($f >= 1.5)}"; then ok=1; break; fi
  echo "$(date -u +%FT%TZ) guard: Windows free ${f} GiB < 1.5; waiting" >> $COPY/diag_out/d2/guard.log
  sleep 30
done
[ "${ok:-0}" = 1 ] || { echo "guard timeout" >> $COPY/diag_out/d2/guard.log; exit 3; }
cd $COPY
export FLYCIRCUIT_HOME=$ORIG PYTHONPATH=$COPY/src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
case "$script" in /*) ;; *) script="$COPY/$script" ;; esac
exec $ORIG/scripts/linux/heavyjob.sh "$label" $ORIG/.venv/bin/python "$script" "$@"
