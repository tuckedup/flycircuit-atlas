#!/usr/bin/env bash
COPY="${FLYCIRCUIT_COPY:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$COPY" || exit 2
OUTD=$COPY/results_v4/learning_readout_v4_1
if pgrep -af "sign_probe|encoder_probe|heavyjob" | grep -v pgrep; then echo "job already running"; exit 3; fi
sha256sum scripts_v4/sign_probe.py scripts_v4/summarize_w2.py scripts_v4/width_probe.py scripts_v4/run_guarded.sh \
  "$OUTD/W2_PREREG.md" "$OUTD/seed_registry.json" configs/experiments/learning_v3_protocol.yaml \
  src/flycircuit/models/sparse_rnn.py src/flycircuit/eval/learning.py src/flycircuit/eval/learning_v2.py > "$OUTD/W2_prelaunch_hashes.txt"
date -u >> "$OUTD/W2_prelaunch_hashes.txt"
exec bash scripts_v4/run_guarded.sh w2-sign-policy scripts_v4/sign_probe.py > "$OUTD/W2_run.log" 2>&1
