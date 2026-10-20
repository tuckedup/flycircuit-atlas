"""FlyGym stages pause between completed models and resume without repeating them."""
from types import SimpleNamespace

import pytest

from flycircuit.bodies import export as policy_export
from flycircuit.bodies import stages as S
from flycircuit.eval import navigation as N


@pytest.fixture
def environment(tmp_path, monkeypatch):
    clock = [0.0]
    calls = []
    monkeypatch.setattr(S, "RES", tmp_path)
    monkeypatch.setattr(S, "CONDITIONS", ["first", "second"])
    monkeypatch.setattr(S, "UNTRAINED", [])
    monkeypatch.setattr(S, "require_frozen", lambda: "fixture")
    monkeypatch.setattr(S, "_graph", lambda: (None, None, None, {}))
    monkeypatch.setattr(S, "load_dataset", lambda split: None)
    monkeypatch.setattr(S, "val_loss", lambda *args: 0.1)
    monkeypatch.setattr(S.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(N, "build_model", lambda *args: (object(), {"param_report": {}}))
    monkeypatch.setattr(S, "proto", lambda: {"training": {"epochs": 40,
                       "lr_search": {"grid": [0.001, 0.01], "epochs": 12, "seed": 0}},
                       "seeds": {"train": [1, 2]}})

    def train(model, data, lr, epochs, seed, cfg, threads):
        calls.append((lr, epochs, seed))
        clock[0] += 2
        return {"diverged": False, "loss": 0.1}

    monkeypatch.setattr(S, "train_one", train)
    monkeypatch.setattr(policy_export, "export_policy", lambda model, path: path.write_text("fixture"))
    return tmp_path, calls


def test_search_slice_retains_trials_and_refuses_premature_selection(environment):
    root, calls = environment
    args = SimpleNamespace(threads=1, max_minutes=1 / 60)
    assert S.cmd_search(args) == 75
    assert len(calls) == 1 and (root / "search.json").exists()
    assert not (root / "best_lr.json").exists()
    args.max_minutes = None
    assert S.cmd_search(args) is None
    assert len(calls) == 4 and (root / "best_lr.json").exists()
    assert all(epochs == 12 for _, epochs, _ in calls)


def test_training_slice_saves_complete_models_then_resumes(environment):
    root, calls = environment
    (root / "best_lr.json").write_text('{"first": {"lr": 0.001}, "second": {"lr": 0.01}}')
    args = SimpleNamespace(threads=1, max_minutes=1 / 60)
    assert S.cmd_train(args) == 75
    assert len(calls) == 1 and len(list((root / "models").glob("*.npz"))) == 1
    args.max_minutes = None
    assert S.cmd_train(args) is None
    assert len(calls) == 4 and len(list((root / "models").glob("*.npz"))) == 4
    assert all(epochs == 40 for _, epochs, _ in calls)


def test_cli_returns_resumable_exit_status(monkeypatch):
    monkeypatch.setattr(S, "cmd_search", lambda args: 75)
    assert S.main(["search", "--max-minutes", "1"]) == 75


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_time_budgets(value):
    with pytest.raises(SystemExit) as error:
        S.main(["train", "--max-minutes", value])
    assert error.value.code == 2
