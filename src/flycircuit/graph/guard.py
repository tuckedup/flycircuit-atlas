"""Central guard against accidental dense adjacency allocation.

Every dense conversion of a graph-sized matrix must go through ``guard_dense``.
Tiny dense oracles (tests) are allowed under an explicit, bounded element count.
"""

from __future__ import annotations

from math import prod

# 2048 x 2048 float32 = 16 MiB. Anything larger must be sparse.
MAX_DENSE_ELEMENTS = 2048 * 2048


class DenseAllocationError(RuntimeError):
    """Raised when a dense allocation exceeds the permitted element count."""


def guard_dense(shape: tuple[int, ...], max_elements: int | None = None, what: str = "array") -> int:
    limit = MAX_DENSE_ELEMENTS if max_elements is None else max_elements
    n = prod(int(s) for s in shape)
    if n > limit:
        raise DenseAllocationError(
            f"refusing dense {what} of shape {tuple(shape)} ({n} elements > limit {limit}); "
            "use a sparse representation"
        )
    return n
