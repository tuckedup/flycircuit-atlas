"""Shared adapter contracts. Never return fabricated data; never mix releases by bare ID.

- ``qualified_id(dataset, version, raw_id)`` is the ONLY external identity. Bare integers from
  different releases are never concatenated.
- ``CrossDatasetMapping`` is an explicit, scoped, confidence-carrying identity link; a boolean flag
  cannot make two datasets' neurons equivalent. ``MappingTable`` rejects bare-ID or same-release links.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field


class MissingPrerequisite(RuntimeError):
    """Access prerequisite (token, optional package, network) is absent. Unrelated work is unaffected."""


class UnsupportedOperation(NotImplementedError):
    """Operation not available for this dataset/release (reason in message). No data is faked."""


def qualified_id(dataset: str, version: str, raw_id: int | str) -> str:
    return f"{dataset}:{version}:{int(raw_id)}"


def split_qualified_id(uid: str) -> tuple[str, str, int]:
    parts = str(uid).split(":")
    if len(parts) != 3 or not parts[2].lstrip("-").isdigit():
        raise ValueError(f"not a release-qualified id (dataset:version:int): {uid!r}")
    return parts[0], parts[1], int(parts[2])


@dataclass(frozen=True)
class CrossDatasetMapping:
    """Explicit, scoped identity link between two releases (a boolean flag is not enough)."""

    source_uid: str
    target_uid: str
    method: str  # e.g. "published cell-type match (Schlegel 2024 Table S?)"
    confidence: float  # 0..1
    scope: str  # biological scope of the claimed equivalence, e.g. "cell type, not single neuron"

    def __post_init__(self):
        s, t = split_qualified_id(self.source_uid), split_qualified_id(self.target_uid)
        if s[:2] == t[:2]:
            raise ValueError("a cross-dataset mapping must link two different dataset releases")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")
        if not self.method.strip() or not self.scope.strip():
            raise ValueError("method and scope are required: equivalence must be justified and scoped")


@dataclass(frozen=True)
class TypeMapping:
    """Cell-TYPE-level link taken from a source's own annotation column. Types are names, not neuron
    identities: this never asserts that two individual neurons are the same cell."""

    source_dataset: str
    source_version: str
    source_type: str
    target_dataset: str
    target_release: str  # may be "unspecified in source annotation"
    target_type: str
    method: str
    confidence: float
    scope: str = "cell type name correspondence only (not single-neuron identity)"

    def __post_init__(self):
        if self.source_dataset == self.target_dataset and self.source_version == self.target_release:
            raise ValueError("type mapping must link two different dataset releases")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")


@dataclass
class MappingTable:
    mappings: list[CrossDatasetMapping] = field(default_factory=list)

    def add(self, m: CrossDatasetMapping) -> None:
        self.mappings.append(m)

    def targets(self, source_uid: str, min_confidence: float = 0.0) -> list[CrossDatasetMapping]:
        split_qualified_id(source_uid)  # bare IDs are rejected
        return [m for m in self.mappings if m.source_uid == source_uid and m.confidence >= min_confidence]


class DatasetAdapter:
    dataset = "?"
    default_version: str | None = None
    access = "unverified"
    #: honest capability flags, surfaced by `describe()`
    capabilities: dict[str, str] = {}

    def __init__(self, version: str | None = None):
        self.version = version or self.default_version

    def describe(self) -> dict:
        return {"dataset": self.dataset, "version": self.version, "access": self.access,
                "capabilities": dict(self.capabilities)}

    def neurons(self):  # pragma: no cover - interface
        raise UnsupportedOperation(f"{self.dataset}: neurons() not available")

    def edge_batches(self) -> Iterator:  # pragma: no cover - interface
        raise UnsupportedOperation(f"{self.dataset}: edge_batches() not available")

    def edges(self):
        raise UnsupportedOperation(
            f"{self.dataset}: edges() would materialise the full graph; iterate edge_batches() "
            "and aggregate with bounded memory")

    def uid(self, raw_id: int | str) -> str:
        return qualified_id(self.dataset, str(self.version), raw_id)
