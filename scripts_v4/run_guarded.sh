#!/usr/bin/env bash
# Usage: run_guarded.sh <label> <python-script> [args...]
# Waits (<=30 min) for >=1.5 GiB free Windows RAM (WSL2 only; skipped with a warning elsewhere), then runs the script.
# Environment: FLYCIRCUIT_COPY (this repo; default = parent of this script), FLYCIRCUIT_HOME (prepared inputs; default = FLYCIRCUIT_COPY),
#              FLYCIRCUIT_PYTHON (interpreter; default python3). If $FLYCIRCUIT_HOME/scripts/linux/heavyjob.sh exists it is used as the
#              single-heavy-job lock wrapper; it is NOT shipped in this repository.
set -u
COPY="${FLYCIRCUIT_COPY:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ORIG="${FLYCIRCUIT_HOME:-$COPY}"
PY="${FLYCIRCUIT_PYTHON:-python3}"
label="$1"; shift
script="$1"; shift
mkdir -p "$COPY/diag_out/d2"
winfree() { powershell.exe -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)" 2>/dev/null | tr -d '\r'; }
if command -v powershell.exe >/dev/null 2>&1; then
  for _ in $(seq 1 60); do
    f=$(winfree); f=${f:-0}
    if awk "BEGIN{exit !($f >= 1.5)}"; then ok=1; break; fi
    echo "$(date -u +%FT%TZ) guard: Windows free ${f} GiB < 1.5; waiting" >> "$COPY/diag_out/d2/guard.log"
    sleep 30
  done
  [ "${ok:-0}" = 1 ] || { echo "guard timeout" >> "$COPY/diag_out/d2/guard.log"; exit 3; }
else
  echo "WARNING: powershell.exe not found; Windows memory guard skipped (not WSL2)." >&2
fi
cd "$COPY"
export FLYCIRCUIT_HOME="$ORIG" PYTHONPATH="$COPY/src" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
case "$script" in /*) ;; *) script="$COPY/$script" ;; esac
if [ -x "$ORIG/scripts/linux/heavyjob.sh" ]; then
  exec "$ORIG/scripts/linux/heavyjob.sh" "$label" "$PY" "$script" "$@"
fi
exec "$PY" "$script" "$@"
