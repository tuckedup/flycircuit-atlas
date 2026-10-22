"""Fixed-capacity, observed-stream replay baselines; no tensors or external imports."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Item:
    event_id: int
    context: int
    cue: tuple[float, ...]
    label: int

    def __post_init__(self):
        if type(self.event_id) is not int or self.event_id < 0:
            raise ValueError("event identity must be a nonnegative integer")
        if type(self.context) is not int or self.context < 0:
            raise ValueError("context must be a nonnegative integer")
        if type(self.label) is not int or self.label not in (0, 1):
            raise ValueError("label must be a binary training contingency")
        if not isinstance(self.cue, tuple) or not self.cue:
            raise ValueError("cue must be a nonempty immutable tuple")
        if not all(type(x) in (int, float) and math.isfinite(x) for x in self.cue):
            raise ValueError("cue must contain finite numeric values")


class Reservoir:
    """Algorithm R: uniform sample of observed TRIAL EVENTS, including repeats."""

    def __init__(self, capacity, seed):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.seen = 0
        self.last_event = -1
        self.items = []
        self.peak_items = 0
        self.peak_serialized_bytes = 0

    def _validate(self, item):
        if not isinstance(item, Item) or item.event_id <= self.last_event:
            raise ValueError(
                "observe requires strictly increasing training-event identities"
            )
        self.last_event = item.event_id
        self.seen += 1

    def _measure(self):
        self.peak_items = max(self.peak_items, len(self.items))
        payload = json.dumps(
            [asdict(x) for x in self.items], separators=(",", ":")
        ).encode()
        self.peak_serialized_bytes = max(self.peak_serialized_bytes, len(payload))

    def observe(self, item):
        self._validate(item)
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            index = self.rng.randrange(self.seen)
            if index < self.capacity:
                self.items[index] = item
        self._measure()

    def sample(self, draw_rng):
        if not self.items:
            raise ValueError("cannot replay an empty buffer")
        return draw_rng.choice(self.items)

    def report(self):
        return {
            "capacity_events": self.capacity,
            "observed_events": self.seen,
            "stored_events": len(self.items),
            "peak_stored_events": self.peak_items,
            "peak_serialized_bytes": self.peak_serialized_bytes,
            "memory_note": "JSON payload bytes exclude Python objects and task dataset storage",
            "contexts": {
                str(c): sum(x.context == c for x in self.items)
                for c in sorted({x.context for x in self.items})
            },
        }


class BalancedReservoir(Reservoir):
    """Uniform within context, nearly equal quotas across OBSERVED contexts.

    Context identity is already observable in the native task. Not applicable to
    a task-free benchmark without declaring this extra information. Capacity must
    permit at least one event per observed context; quota overflow fails explicitly.
    """

    def __init__(self, capacity, seed):
        super().__init__(capacity, seed)
        self.buckets = {}
        self.counts = {}

    def observe(self, item):
        is_new = item.context not in self.buckets
        if is_new and len(self.buckets) == self.capacity:
            raise ValueError(
                "capacity is insufficient to retain every observed context"
            )
        self._validate(item)
        if is_new:
            self.buckets[item.context], self.counts[item.context] = [], 0
        # Arrival order keeps each existing context's quota non-increasing.
        # Sorting numeric IDs could grow a previously shrunk quota, for which
        # discarded historical observations cannot be recovered uniformly.
        contexts = list(self.buckets)
        q, rem = divmod(self.capacity, len(contexts))
        quotas = {c: q + int(i < rem) for i, c in enumerate(contexts)}
        for context in contexts:
            if len(self.buckets[context]) > quotas[context]:
                self.buckets[context] = self.rng.sample(
                    self.buckets[context], quotas[context]
                )
        self.counts[item.context] += 1
        bucket, quota = self.buckets[item.context], quotas[item.context]
        if len(bucket) < quota:
            bucket.append(item)
        else:
            index = self.rng.randrange(self.counts[item.context])
            if index < quota:
                bucket[index] = item
        self.items = [x for c in contexts for x in self.buckets[c]]
        self._measure()

    def sample(self, draw_rng):
        contexts = [c for c in sorted(self.buckets) if self.buckets[c]]
        if not contexts:
            raise ValueError("cannot replay an empty buffer")
        return draw_rng.choice(self.buckets[draw_rng.choice(contexts)])
