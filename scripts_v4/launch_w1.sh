#!/usr/bin/env bash
# Records pre-launch hashes of everything the W1 stage depends on, then launches under the guard.
cd <PROJECT_ROOT> || exit 2
OUTD=<PROJECT_ROOT>/results_v4/learning_readout_v4_1
if pgrep -af "encoder_probe|heavyjob" | grep -v pgrep; then echo "job already running"; exit 3; fi
sha256sum scripts_v4/encoder_probe.py scripts_v4/summarize_w1.py scripts_v4/width_probe.py scripts_v4/run_guarded.sh \
  "$OUTD/ENCODER_DIAGNOSTIC_PLAN.md" "$OUTD/PREREG_WIDTH_PROBE.md" "$OUTD/seed_registry.json" \
  configs/experiments/learning_v3_protocol.yaml src/flycircuit/tasks/association.py \
  src/flycircuit/plasticity/agent.py src/flycircuit/tasks/graph_loader.py > "$OUTD/W1_prelaunch_hashes.txt"
date -u >> "$OUTD/W1_prelaunch_hashes.txt"
exec bash scripts_v4/run_guarded.sh w1-encoder-e8 scripts_v4/encoder_probe.py > "$OUTD/W1_run.log" 2>&1
