"""Export trained torch policies (SparseConnectomeRNN / GRUBaseline / EncoderDecoderOnly) to the
numpy bundle format used by ``np_policy`` inside the isolated FlyGym environment."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def export_policy(model, path: str | Path) -> Path:
    import torch

    path = Path(path)
    kind = getattr(model, "kind", "")
    with torch.no_grad():
        if kind in ("connectome_sparse_rnn", "null_double_edge_swap", "random_sparse_rnn") or hasattr(
                model, "raw_gain"):
            if model.spec.z_dim != 0:
                raise ValueError("export supports z_dim=0 only")
            b = dict(kind="sparse_rnn", n=np.int64(model.n), alpha=np.float64(model.alpha),
                     nonlinearity=np.array(model.spec.nonlinearity),
                     pre=model.pre.cpu().numpy().astype(np.int64),
                     post=model.post.cpu().numpy().astype(np.int64),
                     w=model.edge_values().cpu().numpy().astype(np.float64),
                     bias=model.bias.cpu().numpy().astype(np.float64),
                     encoder=model.encoder.cpu().numpy().astype(np.float64),
                     decoder=model.decoder.cpu().numpy().astype(np.float64),
                     sensory_idx=model.sensory_idx.cpu().numpy().astype(np.int64),
                     readout_idx=model.readout_idx.cpu().numpy().astype(np.int64))
        elif hasattr(model, "cell"):
            c = model.cell
            b = dict(kind="gru", enc_w=model.enc.weight.cpu().numpy().astype(np.float64),
                     enc_b=model.enc.bias.cpu().numpy().astype(np.float64),
                     w_ih=c.weight_ih_l0.cpu().numpy().astype(np.float64),
                     w_hh=c.weight_hh_l0.cpu().numpy().astype(np.float64),
                     b_ih=c.bias_ih_l0.cpu().numpy().astype(np.float64),
                     b_hh=c.bias_hh_l0.cpu().numpy().astype(np.float64),
                     dec_w=model.dec.weight.cpu().numpy().astype(np.float64),
                     dec_b=model.dec.bias.cpu().numpy().astype(np.float64))
        elif hasattr(model, "net"):
            l1, l2 = model.net[0], model.net[2]
            b = dict(kind="encdec", w1=l1.weight.cpu().numpy().astype(np.float64),
                     b1=l1.bias.cpu().numpy().astype(np.float64),
                     w2=l2.weight.cpu().numpy().astype(np.float64),
                     b2=l2.bias.cpu().numpy().astype(np.float64))
        else:
            raise ValueError(f"cannot export model of kind {kind!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **b)
    return path
