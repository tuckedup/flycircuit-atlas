"""Independently addressable random streams.

A single mutable RNG is unsafe for paired designs: an intervention branch that consumes extra
draws would shift every later draw of the baseline branch. Here each *named* stream is derived
from ``SeedSequence([base_seed, run_index, crc32(name)])``, so streams never interact, and two
``RngStreams`` objects built from the same (seed, run) reproduce identical draws per name.
"""

from __future__ import annotations

import zlib

import numpy as np


class RngStreams:
    def __init__(self, base_seed: int, run_index: int):
        self.base_seed = int(base_seed)
        self.run_index = int(run_index)
        self._streams: dict[str, np.random.Generator] = {}

    def stream(self, name: str) -> np.random.Generator:
        if name not in self._streams:
            ss = np.random.SeedSequence([self.base_seed, self.run_index, zlib.crc32(name.encode())])
            self._streams[name] = np.random.default_rng(ss)
        return self._streams[name]

    def fresh(self, name: str) -> np.random.Generator:
        """A new generator for ``name`` at its start state (independent of prior consumption)."""
        self._streams.pop(name, None)
        return self.stream(name)

    def spawn_for_run(self, run_index: int) -> RngStreams:
        return RngStreams(self.base_seed, run_index)
