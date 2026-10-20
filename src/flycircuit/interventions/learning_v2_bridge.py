"""Export frozen learning-v2 acquisition models into the trained causal-screen format.

No training or model selection occurs here. The follow-up screen is exploratory and
uses the same engineered cue task, not a biological assay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from flycircuit.config import project_root
from flycircuit.eval import learning_v2 as V
from flycircuit.graph.signs import SignPolicy
from flycircuit.interventions import trained as T
from flycircuit.interventions.rng import RngStreams

STUDY = "learning-v2 frozen acquisition checkpoints; exploratory follow-up causal screen"


def export_instance(record: dict, proto: dict, base, destination: Path,
                    root: Path | None = None) -> dict:
    """Verify identity, export an owned copy, and check loaded forward-pass parity."""
    root = root or project_root()
    if (record.get("status") != "ok" or record.get("family") != "connectome"
            or record.get("mode") != "std" or record.get("arm") not in {"bptt", "decoder_only"}):
        raise ValueError("only successful standard connectome bptt/decoder_only records are supported")
    expected_hash = V.protocol_hash(proto)
    if record.get("protocol_hash") != expected_hash:
        raise ValueError("record protocol hash does not match the supplied protocol")
    if SignPolicy(**proto["sign_policy"]) != SignPolicy():
        raise ValueError("this bridge supports the v2_default sign policy only")
    source = (root / record["association"]["ckpt_post_acquisition"]).resolve()
    if not source.is_relative_to(root.resolve()) or not source.is_file():
        raise ValueError("source checkpoint is absent or outside the repository")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    blob = torch.load(source, map_location="cpu", weights_only=False)
    meta = blob["meta"]
    expected = {"arm": record["arm"], "family": "connectome", "mode": "std",
                "seed": record["seed"], "task_seed": record["seed"], "lr": record["lr"],
                "protocol_hash": expected_hash, "tag": "post_acq"}
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint metadata does not match the acquisition record")
    if record.get("edge_hash") != base.graph.edge_hash():
        raise ValueError("source record graph hash differs from the selected circuit")
    agent, task, _ = V.load_checkpoint_agent(source, base, {}, proto)
    reference_model = V._sparse_model(proto, base, record["seed"], task.spec.n_in)
    for field in ("pre", "post", "prior_val", "sign_node", "sensory_idx", "readout_idx", "encoder"):
        if not torch.equal(agent.model.state_dict()[field], reference_model.state_dict()[field]):
            raise ValueError(f"checkpoint fixed graph/interface differs: {field}")
    for split in ("val", "test"):
        if agent.evaluate(split, 0) != record["association"][f"{split}_acc_acq"]:
            raise ValueError(f"restored {split} acquisition accuracy differs from the saved record")
    model = agent.model
    payload = {
        "format": T.INSTANCE_FORMAT, "kind": "sparse_rnn", "seed": record["seed"],
        "learner": record["arm"], "lr": record["lr"], "sign_variant": "v2_default",
        "sign_tag": base.notes.get("sign_policy_v2", base.notes.get("sign_policy", "v2_default")),
        "circuit": proto["circuit"]["name"], "min_contacts": proto["circuit"]["min_contacts"],
        "edge_hash": base.graph.edge_hash(), "mapping_hash": base.graph.mapping_hash(),
        "protocol_hash": expected_hash, "proto": proto,
        "model_spec": vars(model.spec),
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "f_mu": agent.f_mu.detach().cpu().clone(), "f_sd": agent.f_sd.detach().cpu().clone(),
        "w_eff": model.edge_values().detach().cpu().clone(),
        "train_summary": {"val_acc": record["association"]["val_acc_acq"],
                          "test_acc": record["association"]["test_acc_acq"],
                          "n_trainable": record["n_params_trainable"],
                          "train_seconds": None,
                          "source_record_elapsed_s": record["elapsed_s"],
                          "n_acquisition_trials": proto["training"]["trials_per_phase"]},
        "origin": STUDY, "source_checkpoint": str(source.relative_to(root.resolve())),
        "source_checkpoint_sha256": source_hash,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    reused = destination.exists()
    if reused:
        existing = torch.load(destination, map_location="cpu", weights_only=False)
        if (existing.get("source_checkpoint_sha256") != source_hash
                or existing.get("protocol_hash") != expected_hash):
            raise ValueError("refusing to overwrite a different exported checkpoint")
    else:
        torch.save(payload, destination)
    instance = T.load_instance(destination)
    cues, _ = T.heldout_cues(task)
    x = torch.as_tensor(task.make_input(cues, 0, np.random.default_rng(0)))
    with torch.no_grad():
        expected_logits, _ = agent._forward(x, False, 0.0, False, None)
        restored_logits = T.branch_logits(instance, x, None, RngStreams(0, 0))
    difference = float((expected_logits - restored_logits).abs().max())
    if not torch.allclose(expected_logits, restored_logits, atol=1e-5, rtol=1e-5):
        raise ValueError("causal-instance forward differs from the learned model")
    return {"path": str(destination), "seed": record["seed"], "arm": record["arm"],
            "source_checkpoint_sha256": source_hash, "forward_max_abs_error": difference,
            "val_acc": record["association"]["val_acc_acq"],
            "test_acc": record["association"]["test_acc_acq"], "reused": reused}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--level", choices=["neuron", "type"], default="neuron")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    root = project_root()
    proto = V.load_protocol()
    from flycircuit.config import load_yaml
    prereg = load_yaml(root / "configs/prereg_v2.yaml")
    if not prereg["frozen"] or prereg["protocol_hash"] != V.protocol_hash(proto):
        raise ValueError("learning-v2 protocol must be frozen and match the preregistration")
    records = V.read_records("confirm")
    selection = json.loads((V.stage_dir("search") / "lr_selected.json").read_text())
    expected = {V._key(dict(arm=a, family=f, mode=m, lr=lr, seed=s))
                for a, f, m, lr, _dlr, s in V.build_jobs(proto, selection, proto["seeds"]["confirmatory"])}
    if not expected <= {V._key(record) for record in records}:
        raise ValueError("final learning stage is incomplete; finish it before follow-up screening")
    base = V.load_base(proto)
    instance_dir = root / "runs/learning_v2_causal_instances"
    exports = []
    for arm in ("bptt", "decoder_only"):
        for seed in proto["seeds"]["confirmatory"]:
            matches = [record for record in records if record["arm"] == arm
                       and record["family"] == "connectome" and record["mode"] == "std"
                       and record["seed"] == seed]
            if len(matches) != 1:
                raise ValueError(f"need exactly one {arm}/seed {seed} record")
            exports.append(export_instance(matches[0], proto, base,
                           instance_dir / "instances/v2_default" / f"{arm}_s{seed}.pt"))
    info = {"study": STUDY, "protocol_hash": V.protocol_hash(proto), "exports": exports,
            "note": "This is an exploratory screen defined after final learning outcomes were available."}
    (instance_dir / "export_manifest.json").write_text(json.dumps(info, indent=2))
    if not args.export_only:
        cfg = {"protocol_version": "screen-learning-v2-followup-1", "circuit": proto["circuit"]["name"],
               "min_contacts": proto["circuit"]["min_contacts"], "learners": ["bptt", "decoder_only"],
               "sign_variants": ["v2_default"], "instance_seeds": proto["seeds"]["confirmatory"],
               "instance_study": STUDY, "instance_protocol_hash": V.protocol_hash(proto),
               "min_baseline_acc": 0.60, "seed": 0, "n_interventions": 120,
               "min_distinct_targets": 100, "level": args.level, "min_degree": 1,
               "exclude_sensory": False, "t_start": 0, "n_controls": 2, "n_rep": 8,
               "torch_threads": args.threads, "device": args.device,
               "instances_run_dir": str(instance_dir),
               "sweep": {"n_targets": 30, "n_rep": 4,
                         "doses": {"clamp_silence": [0.0], "clamp_excite": [0.5, 1.0, 2.0],
                                   "disconnect_both": [0.25, 0.5, 1.0]},
                         "windows": [[0, None], [0, 4], [4, None], [5, None]]}}
        info["screen"] = T.run_trained_study(cfg, stage="screen", use_mlflow=False)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
