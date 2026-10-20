"""The Figure 1 reference must not silently resolve to a same-named example file."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flycircuit.eval.refcheck_feeding import published_counts


def test_reader_uses_exact_figure_member_even_when_legacy_example_exists(tmp_path):
    directory = tmp_path / "data/raw/shiu_published"
    figure = directory / "results/figure_1/data"
    figure.mkdir(parents=True)
    pq.write_table(pa.table({"trial": [0], "flywire_id": [102]}), directory / "sugarR_100Hz.parquet")
    pq.write_table(pa.table({"trial": [0, 0], "flywire_id": [101, 101]}), figure / "sugarR_100Hz.parquet")
    assert np.array_equal(published_counts("sugarR_100Hz", np.array([101, 102]), tmp_path), [[2, 0]])


def test_reader_refuses_ambiguous_legacy_only_file(tmp_path):
    directory = tmp_path / "data/raw/shiu_published"
    directory.mkdir(parents=True)
    (directory / "sugarR_100Hz.parquet").write_bytes(b"legacy example")
    with pytest.raises(FileNotFoundError, match="Figure 1"):
        published_counts("sugarR_100Hz", np.array([101, 102]), tmp_path)
