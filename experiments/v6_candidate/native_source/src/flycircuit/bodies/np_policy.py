"""Pure-numpy policy runtime so trained controllers can run inside the torch-free FlyGym env.

IMPORTANT: this file must import ONLY numpy (it is loaded by file path from the isolated FlyGym
environment). It contains no network access and no LLM calls: actions are a deterministic function
of the observation history and stored weights (see ``ActionLog`` provenance fields).

Bundle format (npz): ``kind`` in {sparse_rnn, gru, encdec} + arrays; exported by bodies/export.py and
parity-tested against the torch modules in tests/unit/test_bodies_np_policy.py.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyPolicy:
    """act(obs[10]) -> a in [-1,1]^2 (tanh of the model output, as in training)."""

    def __init__(self, bundle: dict):
        self.kind = str(np.asarray(bundle["kind"]))
        self.b = {k: np.asarray(v) for k, v in bundle.items() if k != "kind"}
        self.reset()

    @classmethod
    def load(cls, path) -> NumpyPolicy:
        with np.load(path, allow_pickle=False) as z:
            return cls({k: z[k] for k in z.files})

    def reset(self):
        b = self.b
        if self.kind == "sparse_rnn":
            self.h = np.zeros(int(b["n"]))
        elif self.kind == "gru":
            self.h = np.zeros(b["w_hh"].shape[1])
        else:
            self.h = None

    def raw(self, x: np.ndarray) -> np.ndarray:
        """Model output y (before tanh) for one time step."""
        b, x = self.b, np.asarray(x, dtype=np.float64)
        if self.kind == "sparse_rnn":
            n = int(b["n"])
            drive = np.bincount(b["post"], weights=b["w"] * self.h[b["pre"]], minlength=n) + b["bias"]
            inp = x @ b["encoder"]
            np.add.at(drive, b["sensory_idx"], inp)
            phi = {"tanh": np.tanh, "relu": lambda v: np.maximum(v, 0.0),
                   "identity": lambda v: v}[str(np.asarray(b["nonlinearity"]))]
            a = float(b["alpha"])
            self.h = (1 - a) * self.h + a * phi(drive)
            return self.h[b["readout_idx"]] @ b["decoder"]
        if self.kind == "gru":
            e = np.maximum(x @ b["enc_w"].T + b["enc_b"], 0.0)
            hs = b["w_hh"].shape[1]
            gi = e @ b["w_ih"].T + b["b_ih"]
            gh = self.h @ b["w_hh"].T + b["b_hh"]
            r = _sigmoid(gi[:hs] + gh[:hs])
            z = _sigmoid(gi[hs:2 * hs] + gh[hs:2 * hs])
            nn_ = np.tanh(gi[2 * hs:] + r * gh[2 * hs:])
            self.h = (1 - z) * nn_ + z * self.h
            return self.h @ b["dec_w"].T + b["dec_b"]
        if self.kind == "encdec":
            return np.tanh(x @ b["w1"].T + b["b1"]) @ b["w2"].T + b["b2"]
        raise ValueError(f"unknown policy kind {self.kind!r}")

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.tanh(self.raw(obs))
