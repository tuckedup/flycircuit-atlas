#!/usr/bin/env bash
# Regenerate the external proposal diff: new/changed src files vs the pristine snapshot, plus new tests and scripts.
set -u
cd "$(dirname "$0")/.."
out=flycircuit_v4_proposal.diff
diff -ruN --exclude=__pycache__ --exclude='*.pyc' orig_src_snapshot_for_diff src > "$out"
for f in tests/unit/test_v4_*.py scripts_v4/*.py scripts_v4/*.sh; do
  [ -f "$f" ] && diff -uN /dev/null "$f" >> "$out"
done
wc -l "$out"
grep '^+++ ' "$out" | cut -f1
