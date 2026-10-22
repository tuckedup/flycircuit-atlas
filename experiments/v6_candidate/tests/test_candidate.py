import copy
import importlib
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from flycircuit_v6 import comparison, metrics, plan, replay, runner
from flycircuit_v6.__main__ import read_records
from flycircuit_v6.guard import pdelab_jobs, windows_available_gib


def item(i, context=0):
    return replay.Item(i, context, (float(i), 0.0), i % 2)


class MetricTests(unittest.TestCase):
    def test_continual_signed_and_seen_only(self):
        result = metrics.continual([[0.9, 0.1], [0.8, 0.85]])
        self.assertAlmostEqual(result["final_average_accuracy"], 0.825)
        self.assertAlmostEqual(result["mean_seen_task_accuracy"], 0.8625)
        self.assertAlmostEqual(result["forgetting_keep"], 0.1)
        self.assertAlmostEqual(result["backward_transfer"], -0.1)
        self.assertIsNone(result["forward_transfer"])

    def test_negative_forgetting_is_not_absolute_value(self):
        result = metrics.continual([[0.6, 0.1], [0.8, 0.7]])
        self.assertAlmostEqual(result["forgetting_keep"], -0.2)
        self.assertEqual(result["forgetting_clip"], 0)

    def test_matrix_schema(self):
        for bad in (
            [[0.1]],
            [[0.1, 0.2]],
            [[0.1, math.nan], [0.2, 0.4]],
            [[0.1, 0.2], [0.3, 1.1]],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                metrics.continual(bad)

    def test_curve_requires_full_predeclared_grid(self):
        result = metrics.robustness_curve([(1, 0.2), (0, 1), (0.25, 0.8)], [0, 0.25, 1])
        self.assertAlmostEqual(result["robustness_auc"], 0.6)
        self.assertEqual(result["worst_declared_rate_success"], 0.2)
        with self.assertRaises(ValueError):
            metrics.robustness_curve([(0, 1), (0.25, 0.8)], [0, 0.25, 1])
        with self.assertRaises(ValueError):
            metrics.robustness_curve([(0, 1), (0, 1), (1, 0)], [0, 1])

    def test_zero_clean_has_no_normalized_retention(self):
        self.assertIsNone(
            metrics.robustness_curve([(0, 0), (1, 0)], [0, 1])["retained_auc_fraction"]
        )

    def test_spl_requires_real_path_reference(self):
        self.assertEqual(metrics.spl(True, 10, 5), 0.5)
        self.assertEqual(metrics.spl(False, 0, 5), 0)
        with self.assertRaises(ValueError):
            metrics.spl(True, 2, 5)
        with self.assertRaises(ValueError):
            metrics.spl(True, 10, None)

    def test_paired_seed_identity_and_sign(self):
        result = metrics.paired_seed_difference(
            {1: 0.7, 2: 0.6}, {1: 0.8, 2: 0.7}, draws=100
        )
        self.assertAlmostEqual(result["mean_difference"], -0.1)
        self.assertTrue(result["small_n_warning"])
        self.assertLess(result["ci95"][1], 0)
        with self.assertRaises(ValueError):
            metrics.paired_seed_difference({1: 0.9}, {2: 0.8})

    def test_episode_failures_do_not_shrink_denominator(self):
        records = [
            {"episode_id": 1, "status": "ok", "success": True},
            {"episode_id": 2, "status": "timeout", "success": True},
        ]
        self.assertEqual(metrics.episode_success(records, [1, 2])["success_rate"], 0.5)
        with self.assertRaises(ValueError):
            metrics.episode_success(records[:1], [1, 2])
        with self.assertRaises(ValueError):
            metrics.episode_success([records[0], records[0]], [1, 2])


class BufferTests(unittest.TestCase):
    def test_capacity_and_reproducibility(self):
        a, b = replay.Reservoir(7, 8), replay.Reservoir(7, 8)
        for i in range(100):
            a.observe(item(i))
            b.observe(item(i))
        self.assertEqual(a.items, b.items)
        self.assertEqual(len(a.items), 7)
        self.assertEqual(a.peak_items, 7)
        self.assertTrue(any(x.event_id < 50 for x in a.items))

    def test_algorithm_r_does_not_only_keep_first_or_last_events(self):
        early = late = 0
        for seed in range(600):
            buf = replay.Reservoir(2, seed)
            for i in range(20):
                buf.observe(item(i))
            early += any(x.event_id == 0 for x in buf.items)
            late += any(x.event_id == 19 for x in buf.items)
        # Expected 60 each; deterministic broad bounds detect prefix/FIFO bugs.
        self.assertTrue(30 < early < 95, early)
        self.assertTrue(30 < late < 95, late)

    def test_balanced_storage_and_sampling(self):
        buf = replay.BalancedReservoir(6, 7)
        i = 0
        for context, size in ((9, 100), (2, 30), (8, 10)):
            for _ in range(size):
                buf.observe(item(i, context))
                i += 1
        self.assertEqual(buf.report()["contexts"], {"2": 2, "8": 2, "9": 2})
        self.assertEqual(buf.peak_items, 6)
        rng = random.Random(9)
        draws = [buf.sample(rng).context for _ in range(600)]
        self.assertTrue(all(140 < draws.count(c) < 260 for c in (9, 2, 8)))

    def test_observation_identity_and_capacity_fail_closed(self):
        buf = replay.Reservoir(1, 1)
        buf.observe(item(1))
        with self.assertRaises(ValueError):
            buf.observe(item(1))
        balanced = replay.BalancedReservoir(1, 1)
        balanced.observe(item(0, 0))
        with self.assertRaises(ValueError):
            balanced.observe(item(1, 1))
        with self.assertRaises(ValueError):
            replay.Item(1, 0, (math.nan,), 0)


class FakeTask:
    spec = SimpleNamespace(n_ctx=3)

    def sample_train(self, rng, phase, reversed_):
        if reversed_:
            raise RuntimeError("unexpected reversal")
        value = rng.random()
        return [[value, float(phase)]], [int(value > 0.5)]


class FakeAgent:
    def __init__(self):
        self.calls = []

    def trial(self, cue, context, label, noise, action):
        noise.random()
        action.random()
        self.calls.append((context, label))

    def evaluate(self, split, context):
        if split != "val":
            raise RuntimeError("heldout exposure")
        return 0.6 + context * 0.1


def fake_streams(seed):
    def stream(offset):
        return SimpleNamespace(
            order=random.Random(seed + offset),
            noise=random.Random(seed + offset + 1),
            act=random.Random(seed + offset + 2),
        )

    return SimpleNamespace(A=stream(0), B=stream(10), C=stream(20))


class RunnerTests(unittest.TestCase):
    def test_all_conditions_share_current_schedule_and_budget(self):
        schedules = []
        for method in runner.METHODS:
            agent = FakeAgent()
            result = runner.run_continual(
                agent,
                FakeTask(),
                {"training": {"trials_per_phase": 5}},
                42,
                method,
                fake_streams(42),
                lambda x: [list(x)],
                capacity=6,
            )
            schedules.append(result["current_schedule_sha256"])
            self.assertEqual(len(agent.calls), 15 if method == "standard" else 25)
            self.assertLessEqual(result["buffer"].get("peak_stored_events", 0), 6)
            self.assertFalse(result["sota_established"])
        self.assertEqual(len(set(schedules)), 1)

    def test_runtime_interrupt_propagates_without_a_fake_result(self):
        def stop():
            raise InterruptedError("test")

        with self.assertRaises(InterruptedError):
            runner.run_continual(
                FakeAgent(),
                FakeTask(),
                {"training": {"trials_per_phase": 5}},
                1,
                "standard",
                fake_streams(1),
                lambda x: [list(x)],
                stop=stop,
            )

    def test_training_is_lazy(self):
        importlib.import_module("flycircuit_v6.__main__")
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)


class ContractTests(unittest.TestCase):
    def test_incompatible_paper_protocol_cannot_be_ranked(self):
        a = dict.fromkeys(comparison.FIELDS, "same")
        b = dict(a, episode_horizon=600)
        a["episode_horizon"] = 180
        result = comparison.comparability(a, b)
        self.assertFalse(result["protocol_comparable"])
        self.assertIn("episode_horizon", result["mismatches"])
        self.assertFalse(comparison.comparability(a, a)["sota_established"])
        self.assertFalse(comparison.comparability({}, {})["protocol_comparable"])

    def test_resource_pareto_requires_all_metrics(self):
        base = {
            "score": 0.8,
            "trainable_parameters": 100,
            "peak_rss_bytes": 500,
            "optimizer_steps": 50,
        }
        self.assertTrue(
            comparison.resource_dominance(dict(base, score=0.9), base)[
                "point_estimate_dominates"
            ]
        )
        self.assertFalse(
            comparison.resource_dominance(
                dict(base, score=0.9, peak_rss_bytes=600), base
            )["point_estimate_dominates"]
        )
        with self.assertRaises(ValueError):
            comparison.resource_dominance({"score": 0.9}, base)

    def test_windows_guard_cim_error_means_unknown(self):
        with patch(
            "flycircuit_v6.guard.subprocess.run", side_effect=OSError("missing")
        ):
            self.assertIsNone(windows_available_gib())

    def test_detects_pdelab_without_killing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp)
            folder = proc / "321"
            folder.mkdir()
            (folder / "cmdline").write_bytes(b"python\0-m\0pdelab.train\0")
            (folder / "cwd").symlink_to(proc)
            self.assertEqual(pdelab_jobs(proc), [321])

    def test_plan_drift_and_result_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, native = root / "candidate", root / "copy"
            (bundle / "flycircuit_v6").mkdir(parents=True)
            protocol = json.loads(
                (Path(__file__).parents[1] / "protocol.json").read_text()
            )
            (bundle / "protocol.json").write_text(json.dumps(protocol))
            (bundle / "flycircuit_v6/a.py").write_text("# fixture")
            for name in plan.NATIVE_SOURCES:
                path = native / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# fixture")
            choices = native / plan.NATIVE_SOURCES[-2]
            choices.write_text(
                json.dumps(
                    {
                        f: {"core_lr": 0.001, "decoder_lr": 0.002}
                        for f in ("connectome", "gru")
                    }
                )
            )
            for path in plan.input_paths(native):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# fixture")
            saved = plan.prepare(bundle, native)
            self.assertEqual(len(saved["coordinates"]), 64)
            plan.verify_plan(saved)
            result_path = root / "rows.jsonl"
            coord = saved["coordinates"][0]
            row = {
                "key": coord["key"],
                "coordinate": coord,
                "plan_sha256": plan.canonical(saved),
                "status": "ok",
                "acceptance_sha256": "f" * 64,
                "result": result_fixture(coord, saved),
            }
            result_path.write_text(json.dumps(row) + "\n")
            self.assertEqual(len(read_records(result_path, saved, "f" * 64)), 1)
            with result_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                read_records(result_path, saved, "f" * 64)
            (bundle / "flycircuit_v6/a.py").write_text("# changed")
            with self.assertRaises(ValueError):
                plan.verify_plan(saved)

    def test_result_schema_rejects_unmeasured_or_inconsistent_successes(self):
        coord, saved = record_plan_fixture()
        result = result_fixture(coord, saved)
        row = {
            "key": coord["key"],
            "coordinate": coord,
            "plan_sha256": plan.canonical(saved),
            "status": "ok",
            "acceptance_sha256": "f" * 64,
            "result": result,
        }
        mutations = [
            lambda x: x.pop("result"),
            lambda x: x.update(status="pending"),
            lambda x: x["result"].update(method="balanced_replay"),
            lambda x: x["result"].update(seed=True),
            lambda x: x["result"].update(split="test"),
            lambda x: x["result"].update(R_val=[[math.nan] * 3] * 3),
            lambda x: x["result"]["metrics"].update(forgetting_keep=0.0),
            lambda x: x["result"]["trial_counts"].update(current=2399),
            lambda x: x["result"]["actual_adam_steps"].update(min=2399),
            lambda x: x["result"]["params"]["groups"][0].update(lr=0.9),
            lambda x: x["result"].update(native_protocol_file_sha256="b" * 64),
            lambda x: x["result"].update(current_schedule_sha256="missing"),
            lambda x: x["result"].update(peak_rss_mib=0),
            lambda x: x["result"]["buffer"].update(stored_events=1),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(read_records(path, saved, "f" * 64), [row])
            for mutation in mutations:
                altered = copy.deepcopy(row)
                mutation(altered)
                path.write_text(json.dumps(altered) + "\n")
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    read_records(path, saved, "f" * 64)

    def test_failed_status_preserves_error_and_refuses_success_payload(self):
        coord, saved = record_plan_fixture()
        row = {
            "key": coord["key"],
            "coordinate": coord,
            "plan_sha256": plan.canonical(saved),
            "status": "failed",
            "acceptance_sha256": "f" * 64,
            "error": "model error",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(read_records(path, saved, "f" * 64), [row])
            row["result"] = result_fixture(coord, saved)
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                read_records(path, saved, "f" * 64)

    def test_result_rejects_torn_duplicate_fields_and_pair_identity_drift(self):
        coord, saved = record_plan_fixture()
        coord2 = dict(coord, method="extra_current")
        coord2["key"] = plan.canonical({k: v for k, v in coord2.items() if k != "key"})
        saved["coordinates"].append(coord2)
        rows = [
            {
                "key": c["key"],
                "coordinate": c,
                "plan_sha256": plan.canonical(saved),
                "status": "ok",
                "acceptance_sha256": "f" * 64,
                "result": result_fixture(c, saved),
            }
            for c in (coord, coord2)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text("".join(json.dumps(x) + "\n" for x in rows))
            self.assertEqual(len(read_records(path, saved, "f" * 64)), 2)
            rows[1]["result"]["current_schedule_sha256"] = "9" * 64
            path.write_text("".join(json.dumps(x) + "\n" for x in rows))
            with self.assertRaises(ValueError):
                read_records(path, saved, "f" * 64)
            path.write_text(json.dumps(rows[0]))
            with self.assertRaises(ValueError):
                read_records(path, saved, "f" * 64)
            path.write_text('{"status":"ok","status":"failed"}\n')
            with self.assertRaises(ValueError):
                read_records(path, saved, "f" * 64)

    def test_audit_and_resume_require_original_acceptance_binding(self):
        coord, saved = record_plan_fixture()
        row = {
            "key": coord["key"],
            "coordinate": coord,
            "plan_sha256": plan.canonical(saved),
            "acceptance_sha256": "f" * 64,
            "status": "ok",
            "result": result_fixture(coord, saved),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(len(read_records(path, saved, "f" * 64)), 1)
            with self.assertRaisesRegex(ValueError, "exact accepted receipt"):
                read_records(path, saved)
            with self.assertRaisesRegex(ValueError, "provenance"):
                read_records(path, saved, "8" * 64)

    def test_input_closure_and_receipt_inventory_are_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, native = root / "candidate", root / "copy"
            (bundle / "flycircuit_v6").mkdir(parents=True)
            (bundle / "protocol.json").write_bytes(
                (Path(__file__).parents[1] / "protocol.json").read_bytes()
            )
            (bundle / "flycircuit_v6/a.py").write_text("# fixture")
            for name in plan.NATIVE_SOURCES:
                path = native / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# fixture")
            (native / plan.NATIVE_SOURCES[-2]).write_text(
                json.dumps(
                    {
                        f: {"core_lr": 0.001, "decoder_lr": 0.002}
                        for f in ("connectome", "gru")
                    }
                )
            )
            for path in plan.input_paths(native):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# fixture")
            saved = plan.prepare(bundle, native)
            selection = plan.input_paths(native)[-1]
            self.assertIn(str(selection.resolve()), saved["input_sha256"])
            plan_path, receipt_path = root / "plan.json", root / "acceptance.json"
            plan_path.write_text(json.dumps(saved))
            receipt = {
                "verdict": "ship",
                "plan_sha256": plan.digest(plan_path),
                "reviewer": "test fixture only",
                "seed_exposure_review": "passed",
                "native_smoke_test": "passed",
                "input_sha256": saved["input_sha256"],
            }
            receipt_path.write_text(json.dumps(receipt))
            self.assertEqual(
                plan.acceptance(plan_path, receipt_path, with_binding=True),
                (saved, plan.canonical(receipt)),
            )
            changed = copy.deepcopy(receipt)
            changed["input_sha256"][str(selection.resolve())] = "0" * 64
            receipt_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                plan.acceptance(plan_path, receipt_path)
            receipt_path.write_text(json.dumps(receipt))
            selection.write_text("changed canonical selection")
            with self.assertRaises(ValueError):
                plan.verify_plan(saved)


def record_plan_fixture():
    coord = {
        "family": "connectome",
        "method": "standard",
        "seed": 26001,
        "core_lr": 0.001,
        "decoder_lr": 0.002,
        "capacity": 32,
    }
    coord["key"] = plan.canonical(coord)
    saved = {
        "coordinates": [coord],
        "source_root": str(Path("/fixture/copy").resolve()),
        "protocol": {"native_n_contexts": 3, "native_trials_per_phase": 800},
        "source_sha256": {
            str(
                Path(
                    "/fixture/copy/configs/experiments/learning_v3_protocol.yaml"
                ).resolve()
            ): "a" * 64
        },
    }
    return coord, saved


def result_fixture(coord, saved):
    phases, n = (
        saved["protocol"]["native_n_contexts"],
        saved["protocol"]["native_trials_per_phase"],
    )
    replaying = coord["method"] in ("uniform_replay", "balanced_replay")
    extras = (phases - 1) * n
    steps = phases * n + (extras if coord["method"] != "standard" else 0)
    core, decoder = {"connectome": (7399, 152), "gru33": (7392, 68)}[coord["family"]]
    matrix = [[0.9, 0.5, 0.5], [0.8, 0.9, 0.5], [0.7, 0.8, 0.9]]
    buffer = {"capacity_events": coord["capacity"], "stored_events": 0}
    if replaying:
        buffer.update(
            stored_events=coord["capacity"],
            observed_events=phases * n,
            peak_stored_events=coord["capacity"],
            peak_serialized_bytes=1024,
            contexts={"0": coord["capacity"]},
            memory_note="fixture",
        )
    return {
        "method": coord["method"],
        "family": coord["family"],
        "seed": coord["seed"],
        "split": "val",
        "sota_established": False,
        "R_val": matrix,
        "metrics": metrics.continual(matrix),
        "trial_counts": {
            "current": phases * n,
            "replay": extras if replaying else 0,
            "extra_current": extras if coord["method"] == "extra_current" else 0,
            "replay_old_context": extras if replaying else 0,
            "replay_current_context": 0,
        },
        "optimizer_steps_expected": steps,
        "actual_adam_steps": {"min": steps, "max": steps, "n_tensors": 2},
        "native_n_contexts": phases,
        "native_trials_per_phase": n,
        "native_protocol_file_sha256": saved["source_sha256"][
            str(
                (
                    Path(saved["source_root"])
                    / "configs/experiments/learning_v3_protocol.yaml"
                ).resolve()
            )
        ],
        "task_protocol_hash": "a" * 16,
        "w76_sha256": "c" * 64,
        "graph_mapping_sha256": "1" * 64,
        "graph_edges_sha256": "2" * 64,
        "graph_signs_sha256": "3" * 64,
        "training_validation_identity_sha256": "d" * 64,
        "current_schedule_sha256": "e" * 64,
        "params": {
            "optimizer": "Adam",
            "owned_total": core + decoder,
            "requires_grad_total": core + decoder,
            "core_owned": core,
            "decoder_owned": decoder,
            "groups": [
                {"group": 0, "lr": coord["core_lr"], "numel": core, "n_tensors": 1},
                {
                    "group": 1,
                    "lr": coord["decoder_lr"],
                    "numel": decoder,
                    "n_tensors": 1,
                },
            ],
        },
        "model_info": {},
        "optimizer_name": "Adam",
        "torch_version": "fixture-only",
        "elapsed_s": 1.0,
        "peak_rss_mib": 1.0,
        "buffer": buffer,
    }


class GuardTests(unittest.TestCase):
    def test_replacement_receipt_cannot_reuse_or_reset_existing_ledger(self):
        from flycircuit_v6 import guard

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = root / "flycircuit/runs/.heavyjob.lock"
            lock.parent.mkdir(parents=True)
            lock.write_text("")
            plan_path = root / "plan.json"
            plan_path.write_text("{}")
            ledger = root / "resource_ledger.json"
            saved = {
                "charged_wall_s": 50,
                "attempts": 1,
                "inflight": None,
                "plan_sha256": plan.digest(plan_path),
                "acceptance_sha256": "f" * 64,
            }
            ledger.write_text(json.dumps(saved))
            value = {
                "source_root": str(root / "copy"),
                "bundle_root": str(root),
                "protocol": {"wall_limit_s": 1800},
            }
            state = {"pdelab_python_pids": [], "available_windows_gib": 3}
            with (
                patch.object(guard, "acceptance", return_value=(value, "8" * 64)),
                patch.object(guard, "status", return_value=state),
                patch.object(guard.subprocess, "Popen") as popen,
            ):
                with self.assertRaisesRegex(ValueError, "invalid budget identity"):
                    guard.run(plan_path, root / "acceptance.json")
                popen.assert_not_called()
            self.assertEqual(json.loads(ledger.read_text()), saved)

    def test_newer_ledger_written_before_lock_is_never_overwritten(self):
        from flycircuit_v6 import guard

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = root / "flycircuit/runs/.heavyjob.lock"
            lock.parent.mkdir(parents=True)
            lock.write_text("")
            plan_path = root / "plan.json"
            plan_path.write_text("{}")
            ledger = root / "resource_ledger.json"
            old = {
                "charged_wall_s": 0,
                "attempts": 0,
                "inflight": None,
                "plan_sha256": plan.digest(plan_path),
                "acceptance_sha256": "f" * 64,
            }
            ledger.write_text(json.dumps(old))
            newest = dict(old, charged_wall_s=1800, attempts=1)

            def other_attempt_finished(*_):
                ledger.write_text(json.dumps(newest))

            value = {
                "source_root": str(root / "copy"),
                "bundle_root": str(root),
                "protocol": {"wall_limit_s": 1800},
            }
            state = {"pdelab_python_pids": [], "available_windows_gib": 3}
            with (
                patch.object(guard, "acceptance", return_value=(value, "f" * 64)),
                patch.object(guard, "status", return_value=state),
                patch("fcntl.flock", side_effect=other_attempt_finished),
                patch.object(guard.subprocess, "Popen") as popen,
            ):
                with self.assertRaisesRegex(ValueError, "exhausted"):
                    guard.run(plan_path, root / "acceptance.json")
                popen.assert_not_called()
            self.assertEqual(json.loads(ledger.read_text()), newest)

    def test_active_pdelab_prevents_even_input_loading(self):
        from flycircuit_v6 import guard

        with (
            patch.object(
                guard,
                "status",
                return_value={"pdelab_python_pids": [123], "available_windows_gib": 4},
            ),
            patch.object(guard, "acceptance") as accept,
        ):
            with self.assertRaises(ValueError):
                guard.run("plan.json", "acceptance.json")
            accept.assert_not_called()

    def test_memory_abort_and_persisted_wall_budget(self):
        self._run_fake_guard("memory_guard", 1800, 0.5)

    def test_wall_limit_is_not_reset_on_resume(self):
        self._run_fake_guard("cumulative_wall_limit", 40, 3)

    def test_source_drift_terminates_own_child(self):
        self._run_fake_guard("supervisor_error:ValueError", 1800, 3, drift=True)

    def _run_fake_guard(self, expected_reason, cap, runtime_free, drift=False):
        from flycircuit_v6 import guard

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "copy"
            native.mkdir()
            lock = root / "flycircuit/runs/.heavyjob.lock"
            lock.parent.mkdir(parents=True)
            lock.write_text("")
            plan_path = root / "plan.json"
            plan_path.write_text("{}")
            plan_value = {
                "source_root": str(native),
                "bundle_root": str(root),
                "protocol": {"wall_limit_s": cap},
            }
            clock = [0.0]

            class Child:
                pid = 987654321
                returncode = None

                def poll(self):
                    return self.returncode

                def wait(self, timeout=None):
                    self.returncode = -15
                    return self.returncode

            def sleep(seconds):
                clock[0] += seconds

            child = Child()
            state = {"available_windows_gib": 3.0, "pdelab_python_pids": []}
            checks = [None, ValueError("drift")] if drift else None
            with (
                patch.object(guard, "acceptance", return_value=(plan_value, "f" * 64)),
                patch.object(guard, "status", return_value=state),
                patch.object(guard, "verify_plan", side_effect=checks) as verify,
                patch.object(guard, "windows_available_gib", return_value=runtime_free),
                patch.object(guard, "pdelab_jobs", return_value=[]),
                patch.object(guard.time, "monotonic", side_effect=lambda: clock[0]),
                patch.object(guard.time, "sleep", side_effect=sleep),
                patch.object(guard.subprocess, "Popen", return_value=child) as popen,
                patch.object(guard.os, "killpg") as kill,
            ):
                if drift:
                    with self.assertRaises(ValueError):
                        guard.run(plan_path, root / "acceptance.json")
                else:
                    self.assertEqual(guard.run(plan_path, root / "acceptance.json"), 4)
                self.assertTrue(verify.called)
                popen.assert_called_once()
                self.assertEqual(kill.call_args[0][0], child.pid)
            budget = json.loads((root / "resource_ledger.json").read_text())
            self.assertEqual(budget["last_result"]["reason"], expected_reason)
            self.assertIsNone(budget["inflight"])
            self.assertGreaterEqual(budget["charged_wall_s"], 30)
            if cap == 40:
                with (
                    patch.object(
                        guard, "acceptance", return_value=(plan_value, "f" * 64)
                    ),
                    patch.object(guard, "status", return_value=state),
                    patch.object(guard.subprocess, "Popen") as popen,
                ):
                    with self.assertRaises(ValueError):
                        guard.run(plan_path, root / "acceptance.json")
                    popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
