"""Cue-valence association / reversal / context-dependent continual-learning task (SYNTHETIC task).

Engineered task, not a biological assay. A *cue* is a k-sparse binary pattern over ``n_cue`` input
channels (stand-in for an odor code). Each *context* c has a teacher vector ``u_c``; the rewarded
action for cue x in context c is ``approach`` (1) if ``u_c . (x - k/n) > 0`` else ``avoid`` (0).
Teachers of different contexts are partially correlated (``context_corr``) so tasks overlap but
conflict on some cues. Context identity is OBSERVABLE: an extra one-hot block of ``n_ctx`` channels.
Reversal flips the sign of the teacher (same cues, opposite contingency).

Trial timing (``dt`` of the neural clock): T steps; cue channels on for steps [0, cue_steps); context
channels on for all T steps; the action is read from the MEAN readout over steps [T-resp_steps, T);
reward is delivered after the trial (delay = 0 further steps; one decision per trial; terminal).

Leakage control: cue sets for train / val / test come from independent ``SeedSequence`` streams
keyed by (seed, split, context) and are verified pairwise disjoint (pattern hashes). Each
(split, context) set is class balanced for that context's contingency. Hyper-parameter selection
uses ONLY ``val``; ``test`` is touched by final evaluation only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SPLITS = ("train", "val", "test")
_SPLIT_ID = {"train": 0, "val": 1, "test": 2}


@dataclass
class TaskSpec:
    n_cue: int = 32
    k_active: int = 8
    n_ctx: int = 3
    T: int = 8
    cue_steps: int = 6
    resp_steps: int = 3
    input_noise: float = 0.05
    context_corr: float = 0.7
    n_train: int = 200
    n_val: int = 64
    n_test: int = 64

    @property
    def n_in(self) -> int:
        return self.n_cue + self.n_ctx


def _pattern_key(x: np.ndarray) -> bytes:
    return np.packbits(x.astype(np.uint8)).tobytes()


@dataclass
class ContingencyTask:
    spec: TaskSpec
    seed: int
    teachers: np.ndarray = field(init=False)
    cues: dict = field(init=False)  # (split, ctx) -> (n, n_cue) float32 patterns
    labels: dict = field(init=False)  # (split, ctx) -> (n,) int, 1 = approach (base contingency)

    def __post_init__(self):
        s = self.spec
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 99]))
        base = rng.standard_normal(s.n_cue)
        c = s.context_corr
        self.teachers = np.stack(
            [c * base + np.sqrt(1 - c * c) * rng.standard_normal(s.n_cue) for _ in range(s.n_ctx)]
        )
        self.cues, self.labels = {}, {}
        used: set[bytes] = set()
        sizes = {"train": s.n_train, "val": s.n_val, "test": s.n_test}
        for split in SPLITS:
            for ctx in range(s.n_ctx):
                x, y = self._balanced_set(split, ctx, sizes[split], used)
                self.cues[(split, ctx)], self.labels[(split, ctx)] = x, y
        self.assert_no_leakage()

    def _label(self, x: np.ndarray, ctx: int, reversed_: bool = False) -> np.ndarray:
        score = (x - self.spec.k_active / self.spec.n_cue) @ self.teachers[ctx]
        y = (score > 0).astype(np.int64)
        return 1 - y if reversed_ else y

    def _balanced_set(self, split: str, ctx: int, n: int, used: set[bytes]):
        s = self.spec
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, _SPLIT_ID[split], ctx]))
        got = {0: [], 1: []}
        half = n // 2
        while len(got[0]) < half or len(got[1]) < n - half:
            idx = np.stack([rng.permutation(s.n_cue)[: s.k_active] for _ in range(256)])
            x = np.zeros((256, s.n_cue), dtype=np.float32)
            np.put_along_axis(x, idx, 1.0, axis=1)
            y = self._label(x, ctx)
            for xi, yi in zip(x, y, strict=True):
                key = _pattern_key(xi)
                if key in used:
                    continue  # unique across ALL splits and contexts
                want = half if yi == 0 else n - half
                if len(got[int(yi)]) < want:
                    got[int(yi)].append(xi)
                    used.add(key)
        xs = np.stack(got[0] + got[1])
        ys = np.array([0] * len(got[0]) + [1] * len(got[1]), dtype=np.int64)
        perm = rng.permutation(len(ys))
        return xs[perm], ys[perm]

    def assert_no_leakage(self) -> None:
        seen: dict[bytes, str] = {}
        for (split, ctx), x in self.cues.items():
            for xi in x:
                k = _pattern_key(xi)
                if k in seen:
                    raise AssertionError(f"cue shared by {seen[k]} and {split}/{ctx}")
                seen[k] = f"{split}/{ctx}"

    # ---- trial tensors -----------------------------------------------------------------
    def labels_for(self, split: str, ctx: int, reversed_: bool = False) -> np.ndarray:
        y = self.labels[(split, ctx)]
        return 1 - y if reversed_ else y

    def make_input(self, cue: np.ndarray, ctx: int, noise_rng: np.random.Generator | None):
        """cue: (B, n_cue) -> x_seq (T, B, n_in) float32."""
        s = self.spec
        b = cue.shape[0]
        x = np.zeros((s.T, b, s.n_in), dtype=np.float32)
        x[: s.cue_steps, :, : s.n_cue] = cue[None]
        x[:, :, s.n_cue + ctx] = 1.0
        if noise_rng is not None and s.input_noise > 0:
            x += (s.input_noise * noise_rng.standard_normal(x.shape)).astype(np.float32)
        return x

    def response_window(self) -> slice:
        return slice(self.spec.T - self.spec.resp_steps, self.spec.T)

    def sample_train(self, rng: np.random.Generator, ctx: int, reversed_: bool = False):
        """One training trial: returns (cue (1,n_cue), label (1,))."""
        n = self.cues[("train", ctx)].shape[0]
        i = int(rng.integers(n))
        return self.cues[("train", ctx)][i : i + 1], self.labels_for("train", ctx, reversed_)[i : i + 1]
