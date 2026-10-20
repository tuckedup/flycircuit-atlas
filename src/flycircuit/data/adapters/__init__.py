"""Secondary-dataset adapters (MaleCNS, BANC, hemibrain, larva). See docs/DATA_SECONDARY.md.

Every adapter uses release-qualified IDs (``dataset:version:id``); cross-dataset links are explicit
``CrossDatasetMapping`` (ID-level, scoped, confidence) or ``TypeMapping`` (type-name level) objects.
"""

from .banc import BANCAdapter
from .base import (
    CrossDatasetMapping,
    DatasetAdapter,
    MappingTable,
    MissingPrerequisite,
    TypeMapping,
    UnsupportedOperation,
    qualified_id,
    split_qualified_id,
)
from .hemibrain import HemibrainAdapter
from .larva import LarvaAdapter
from .malecns import MaleCNSAdapter

ADAPTERS = {c.dataset: c for c in (MaleCNSAdapter, BANCAdapter, HemibrainAdapter, LarvaAdapter)}
ALIASES = {"larval": "larva"}


def get_adapter(dataset: str, version: str | None = None, **kw) -> DatasetAdapter:
    dataset = ALIASES.get(dataset, dataset)
    if dataset not in ADAPTERS:
        raise UnsupportedOperation(f"unknown dataset {dataset!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[dataset](version, **kw)


__all__ = ["ADAPTERS", "BANCAdapter", "CrossDatasetMapping", "DatasetAdapter", "HemibrainAdapter",
           "LarvaAdapter", "MaleCNSAdapter", "MappingTable", "MissingPrerequisite", "TypeMapping",
           "UnsupportedOperation", "get_adapter", "qualified_id", "split_qualified_id"]
