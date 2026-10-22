"""PROPOSAL / PROFILING COPY of bodies.stages.train_one (original is NOT edited).

Same math as the original (per-sequence dropout masks, TBPTT chunks, masked tanh-action MSE, Adam,
clip_grad_norm_) with (a) phase timers (data conversion, batch/mask building, forward+loss, backward+clip,
optimizer step), (b) counters for padded vs real ticks actually pushed through the recurrent model, and
(c) explicit device plumbing (model, X/Y/M, masks and recurrent state on ``device``).
Used only by scripts for the micro-profile; never produces accepted scientific records.
"""
from __future__ import annotations

import time

import numpy as np


def _sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def train_instrumented(model, data, lr: float, epochs: int, seed: int, tcfg: dict, threads: int = 2,
                       device: str = "cpu", warmup_epochs: int = 1) -> dict:
    import torch

    from flycircuit.models.train import detach_state
    dev = torch.device(device)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    clock = time.perf_counter
    t = {"data_to_device": 0.0, "batch_build": 0.0, "forward_loss": 0.0, "backward_clip": 0.0,
         "opt_step": 0.0}

    _sync(torch, dev)
    t0 = clock()
    model.to(dev)
    X, Y, M = (torch.as_tensor(a).to(dev) for a in data)
    _sync(torch, dev)
    one_time_data = clock() - t0

    levels = np.array(tcfg["dropout_levels_in_training"])
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = np.random.default_rng(seed * 101 + 7)
    n = X.shape[1]
    ticks_proc = ticks_real = n_chunks = n_batches = 0
    losses = []
    epoch_walls = []
    for ep in range(warmup_epochs + epochs):
        timed = ep >= warmup_epochs
        _sync(torch, dev)
        e0 = clock()
        perm = rng.permutation(n)
        for b0 in range(0, n, tcfg["batch_sequences"]):
            _sync(torch, dev)
            c0 = clock()
            idx = perm[b0:b0 + tcfg["batch_sequences"]]
            lv = rng.choice(levels, size=len(idx))
            mask = torch.as_tensor((rng.random((len(idx), 10)) < lv[:, None]).astype(np.float32)).to(dev)
            xb, yb, mb = X[:, idx] * (1 - mask)[None], Y[:, idx], M[:, idx]
            state = None
            _sync(torch, dev)
            if timed:
                t["batch_build"] += clock() - c0
                n_batches += 1
            for s in range(0, X.shape[0], tcfg["tbptt_steps"]):
                xc, yc, mc = (a[s:s + tcfg["tbptt_steps"]] for a in (xb, yb, mb))
                real = float(mc.sum())
                if real == 0:
                    break
                opt.zero_grad(set_to_none=True)
                _sync(torch, dev)
                f0 = clock()
                out, state = model(xc, state)
                loss = ((((torch.tanh(out) - yc) ** 2).sum(-1)) * mc).sum() / mc.sum()
                _sync(torch, dev)
                f1 = clock()
                if not torch.isfinite(loss):
                    return {"diverged": True}
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                _sync(torch, dev)
                f2 = clock()
                opt.step()
                _sync(torch, dev)
                f3 = clock()
                state = detach_state(state)
                if timed:
                    t["forward_loss"] += f1 - f0
                    t["backward_clip"] += f2 - f1
                    t["opt_step"] += f3 - f2
                    ticks_proc += int(mc.numel())
                    ticks_real += int(real)
                    n_chunks += 1
                    losses.append(float(loss.detach()))
        _sync(torch, dev)
        if timed:
            epoch_walls.append(clock() - e0)
    wall = float(sum(epoch_walls))
    return {"diverged": False, "device": device, "threads": threads, "n_sequences": int(n),
            "timed_epochs": epochs, "wall_s_per_epoch": wall / max(epochs, 1),
            "phase_s_per_epoch": {k: v / max(epochs, 1) for k, v in t.items()},
            "one_time_model_and_data_to_device_s": one_time_data,
            "padded_tick_fraction_realized": 1.0 - ticks_real / max(ticks_proc, 1),
            "chunks_per_epoch": n_chunks / max(epochs, 1), "batches_per_epoch": n_batches / max(epochs, 1),
            "loss_last20": float(np.mean(losses[-20:])) if losses else float("nan")}
