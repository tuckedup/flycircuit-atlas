"""Published files with the same basename must retain distinct source identities."""
import hashlib
import io
import zipfile
from types import SimpleNamespace

import pytest

from flycircuit.eval import refcheck_remote as R


@pytest.fixture
def archive(monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as stream:
        stream.writestr("figure/data/same.parquet", b"figure output")
        stream.writestr("example/same.parquet", b"example output")
    data = buffer.getvalue()
    monkeypatch.setattr(R, "open_remote_zip", lambda url: (
        zipfile.ZipFile(io.BytesIO(data)), SimpleNamespace(bytes_fetched=len(data))))


def test_flattened_collisions_fail_before_network_or_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "open_remote_zip", lambda url: pytest.fail("must reject before fetching"))
    with pytest.raises(ValueError, match="collision"):
        R.extract_members("fixture", ["figure/data/same.parquet", "example/same.parquet"], tmp_path)
    assert not list(tmp_path.iterdir())


def test_preserved_paths_keep_distinct_contents_and_hashes(tmp_path, archive):
    result = R.extract_members("fixture", ["figure/data/same.parquet", "example/same.parquet"],
                               tmp_path, preserve_paths=True)
    for member, content in [("figure/data/same.parquet", b"figure output"),
                            ("example/same.parquet", b"example output")]:
        assert (tmp_path / member).read_bytes() == content
        assert result[member]["sha256"] == hashlib.sha256(content).hexdigest()
    reused = R.extract_members("fixture", ["figure/data/same.parquet"], tmp_path, preserve_paths=True)
    assert reused["figure/data/same.parquet"]["reused"]


def test_existing_different_raw_file_is_preserved(tmp_path, archive):
    path = tmp_path / "same.parquet"
    path.write_bytes(b"historical raw file")
    with pytest.raises(ValueError, match="refusing overwrite"):
        R.extract_members("fixture", ["figure/data/same.parquet"], tmp_path)
    assert path.read_bytes() == b"historical raw file"


@pytest.mark.parametrize("member", ["../outside", "/absolute", "C:/outside", "nested/../../outside"])
def test_unsafe_member_paths_are_rejected_before_fetch(tmp_path, monkeypatch, member):
    monkeypatch.setattr(R, "open_remote_zip", lambda url: pytest.fail("must reject before fetching"))
    with pytest.raises(ValueError, match="unsafe"):
        R.extract_members("fixture", [member], tmp_path, preserve_paths=True)
