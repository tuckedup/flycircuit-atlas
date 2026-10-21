"""The release headline verifier must pass on the shipped files and fail on duplicated/missing rows, wrong values and doc drift."""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("verify_w3_headline", ROOT / "scripts_release" / "verify_w3_headline.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

ROWS = "results_v4/learning_readout_v4_1/W3_trained_w76_rows.jsonl"


@pytest.fixture()
def copy(tmp_path):
    for rel in ("results_v4/learning_readout_v4_1", "README.md", "RESULTS_REPORT.md", "ROADMAP.md"):
        src = ROOT / rel
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst) if src.is_dir() else shutil.copy(src, dst)
    return tmp_path


def test_shipped_files_pass():
    fails, facts = V.verify(ROOT)
    assert fails == []
    assert facts["best_seed"] == 1502 and abs(facts["best"] - 0.890625) < 1e-12


def test_duplicate_row_detected(copy):
    p = copy / ROWS
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines + [lines[10]]) + "\n")
    fails, _ = V.verify(copy)
    assert any("duplicate row key" in f for f in fails) and any("row count" in f for f in fails)


def test_missing_row_detected(copy):
    p = copy / ROWS
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:-1]) + "\n")
    fails, _ = V.verify(copy)
    assert any("row count 279" in f for f in fails)


def test_wrong_reported_value_detected(copy):
    p = copy / "README.md"
    p.write_text(p.read_text().replace("89.06 %", "90.06 %"))
    fails, _ = V.verify(copy)
    assert any("89.06" in f for f in fails)


def test_report_table_drift_detected(copy):
    p = copy / "RESULTS_REPORT.md"
    p.write_text(p.read_text().replace("0.824 (0.035)", "0.834 (0.035)"))
    fails, _ = V.verify(copy)
    assert any("acquisition cell" in f for f in fails)


def test_retracted_wording_detected(copy):
    p = copy / "ROADMAP.md"
    p.write_text(p.read_text() + "\nCompared against a matched GRU.\n")
    fails, _ = V.verify(copy)
    assert any("retracted wording" in f for f in fails)
