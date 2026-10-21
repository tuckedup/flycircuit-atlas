#!/usr/bin/env bash
COPY="${FLYCIRCUIT_COPY:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$COPY" || exit 2
OUTD=$COPY/results_v4/learning_readout_v4_1
if pgrep -af "w3_pilot|sign_probe|encoder_probe|heavyjob" | grep -v pgrep; then echo "job already running"; exit 3; fi
sha256sum scripts_v4/w3_pilot.py scripts_v4/summarize_w3.py scripts_v4/run_guarded.sh "$OUTD/W3_PREREG.md" "$OUTD/seed_registry.json" \
  src/flycircuit/eval/learning_v4.py src/flycircuit/eval/learning_v4_select.py configs/experiments/learning_v3_protocol.yaml > "$OUTD/W3_prelaunch_hashes.txt"
date -u >> "$OUTD/W3_prelaunch_hashes.txt"
exec bash scripts_v4/run_guarded.sh W3 $COPY/scripts_v4/w3_pilot.py > "$OUTD/W3_run.log" 2>&1
