"""Configurable roots for the scripts in this directory.

FLYCIRCUIT_COPY : directory holding this repository (default: the repository containing this file).
FLYCIRCUIT_HOME : directory holding the prepared inputs (data/, results/nulls_v2, results/learning_v3, configs/).
                  Default: the same directory as FLYCIRCUIT_COPY, i.e. a single-root layout.
"""
import os
from pathlib import Path

ROOT = os.environ.get("FLYCIRCUIT_COPY") or str(Path(__file__).resolve().parents[1])
CANON = os.environ.get("FLYCIRCUIT_HOME") or ROOT
