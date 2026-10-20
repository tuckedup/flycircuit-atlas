import pytest
from pydantic import ValidationError

from flycircuit.config import HardwareProfile, config_hash, load_hardware
from flycircuit.resources.estimate import estimate_sparse_rnn, preflight

PROBE_OK = {
    "wsl_ram_available_gib": 6.0,
    "cuda": {"available": True, "free_gib": 5.4},
    "host_disk": {"free_gib": 130.0},
}


def test_hardware_profile_matches_spec():
    hw = load_hardware()
    assert hw.cpu_threads == 4 and hw.max_pipeline_ram_gib == 4.5
    assert hw.max_cuda_allocated_gib == 5.5 and hw.whole_brain_gradient_training is False
    assert hw.full_synapse_download_default is False


def test_hardware_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        HardwareProfile(**{**load_hardware().model_dump(), "bogus": 1})


def test_estimate_scales_with_tbptt_and_edges():
    a = estimate_sparse_rnn(1000, 50_000, tbptt=32)
    b = estimate_sparse_rnn(1000, 50_000, tbptt=64)
    c = estimate_sparse_rnn(1000, 100_000, tbptt=32)
    assert b["cuda_bytes"] > a["cuda_bytes"] and c["cuda_bytes"] > a["cuda_bytes"]
    assert a["kind"] == "theoretical"


def test_preflight_passes_small_and_fails_big():
    hw = load_hardware()
    assert preflight(estimate_sparse_rnn(1000, 50_000), hw, PROBE_OK).ok
    big = preflight(estimate_sparse_rnn(140_000, 50_000_000), hw, PROBE_OK)
    assert not big.ok and any("CUDA" in f for f in big.failures)


def test_preflight_disk_and_storage_and_walltime():
    hw = load_hardware()
    est = estimate_sparse_rnn(100, 1000)
    low_disk = {**PROBE_OK, "host_disk": {"free_gib": 45.0}}
    assert not preflight(est, hw, low_disk, new_storage_gib=10).ok
    assert not preflight(est, hw, PROBE_OK, new_storage_gib=31).ok
    assert not preflight(est, hw, PROBE_OK, walltime_minutes=61).ok


def test_config_hash_deterministic():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
