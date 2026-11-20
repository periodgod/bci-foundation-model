import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from eeg_mime_foundation.data import (Batch, ManifestDataset, Normalizer, Sample, assign_grouped_splits,
                                      audit_splits, collate, make_synthetic)
from eeg_mime_foundation.decision import decide
from eeg_mime_foundation.engine import Runner, evaluate, load_checkpoint
from eeg_mime_foundation.fewshot import support_manifest
from eeg_mime_foundation.metrics import metrics
from eeg_mime_foundation.model import ModelConfig, SignalModel, patchify
from eeg_mime_foundation.objectives import (MultiTeacherLoss, band_targets, logit_distillation,
                                           make_mask, reconstruction_loss)
from eeg_mime_foundation.smoke import base_config
from eeg_mime_foundation.stream import CausalStream


def sample(channels=3, samples=103, index=0):
    return Sample(torch.randn(channels, samples), torch.arange(channels), torch.randn(channels, 3) * .03,
                  torch.ones(channels, dtype=torch.bool), torch.tensor(index % 2),
                  str(index), str(index), "0", "0", "test_synthetic")


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        self.batch = collate([sample(), sample(2, 71, 1)])

    def model(self, architecture="fusion"):
        return SignalModel(ModelConfig(architecture=architecture, dim=24, depth=1, heads=4,
                                       local_patches=2, latent_channels=2)).eval()

    def test_all_architectures_ignore_padding_and_permute_channels(self):
        for architecture in ("fusion", "factorized", "compact", "latent", "hierarchical", "causal"):
            with self.subTest(architecture=architecture):
                model = self.model(architecture)
                batch = copy.deepcopy(self.batch)
                original = model(batch)["logits"]
                batch.x = torch.nn.functional.pad(batch.x, (0, 32, 0, 1), value=float("nan"))
                batch.valid = torch.nn.functional.pad(batch.valid, (0, 32, 0, 1), value=False)
                batch.channel_ids = torch.nn.functional.pad(batch.channel_ids, (0, 1), value=-1)
                batch.coords = torch.nn.functional.pad(batch.coords, (0, 0, 0, 1), value=0)
                batch.coord_valid = torch.nn.functional.pad(batch.coord_valid, (0, 1), value=False)
                out = model(batch)
                torch.testing.assert_close(original, out["logits"], atol=2e-6, rtol=2e-6)
                self.assertTrue(torch.equal(out["tokens"][~out["valid"]], torch.zeros_like(out["tokens"][~out["valid"]])))
                order = torch.tensor([2, 0, 3, 1])
                permuted = Batch(batch.x[:, order], batch.valid[:, order], batch.channel_ids[:, order],
                                 batch.coords[:, order], batch.coord_valid[:, order], batch.target, batch.metadata)
                torch.testing.assert_close(out["logits"], model(permuted)["logits"], atol=2e-6, rtol=2e-6)

    def test_masked_content_cannot_leak_and_positions_remain(self):
        model = self.model()
        _, v = patchify(self.batch, 32)
        mask = v.any(-1)
        out = model(self.batch, mask)
        modified = copy.deepcopy(self.batch)
        modified.x += 1000
        torch.testing.assert_close(out["tokens"], model(modified, mask)["tokens"])
        self.assertFalse(torch.allclose(out["tokens"][0, 0, 0], out["tokens"][0, 1, 0]))
        self.assertFalse(torch.allclose(out["tokens"][0, 0, 0], out["tokens"][0, 0, 1]))

    def test_all_invalid_rejected(self):
        self.batch.valid[0] = False
        with self.assertRaisesRegex(ValueError, "All-invalid"):
            self.model()(self.batch)

    def test_loss_ignores_tail_padding(self):
        model = self.model()
        _, v = patchify(self.batch, 32)
        mask = make_mask(v.any(-1), .5)
        out = model(self.batch, mask)
        first, _ = reconstruction_loss(out, mask, model.cfg)
        out["raw"][~out["sample_valid"]] = 1e6
        second, _ = reconstruction_loss(out, mask, model.cfg)
        torch.testing.assert_close(first, second)

    def test_structured_masks_have_targets_and_visible_tokens(self):
        valid = patchify(self.batch, 32)[1].any(-1)
        for kind in ("random", "channel", "time", "joint"):
            mask = make_mask(valid, .95, kind)
            self.assertFalse((mask & ~valid).any())
            self.assertTrue(mask.flatten(1).any(1).all())
            self.assertTrue((valid & ~mask).flatten(1).any(1).all())

    def test_causal_future_independence_and_streaming_equivalence(self):
        model = self.model("causal")
        b = collate([sample(samples=128)])
        first = model(b)["sequence_logits"]
        changed = copy.deepcopy(b)
        changed.x[..., 64:] += 1e4
        torch.testing.assert_close(first[:, :2], model(changed)["sequence_logits"][:, :2])
        stream = CausalStream(model, Normalizer())
        outputs = []
        for lo, hi in ((0, 19), (19, 57), (57, 128)):
            outputs.append(stream.push(b.x[0, :, lo:hi], b.channel_ids[0], b.coords[0], b.coord_valid[0]))
        torch.testing.assert_close(first[0], torch.cat(outputs), atol=2e-6, rtol=2e-6)
        stream.reset()
        again = stream.push(b.x[0], b.channel_ids[0], b.coords[0], b.coord_valid[0])
        torch.testing.assert_close(first[0], again)

    def test_frequency_resolution_rejected(self):
        raw = torch.zeros(1, 1, 1, 16)
        with self.assertRaisesRegex(ValueError, "cannot resolve"):
            band_targets(raw, torch.ones_like(raw, dtype=torch.bool), 64, [(1, 4)])

    def test_tuning_freezes_expected_parameters(self):
        model = self.model()
        model.configure_tuning("linear")
        self.assertEqual({n for n, p in model.named_parameters() if p.requires_grad}, {"head.weight", "head.bias"})
        model.configure_tuning("adapter")
        self.assertTrue(all(n.startswith(("head.", "encoder.adapter.")) for n, p in model.named_parameters() if p.requires_grad))

    def test_variable_sequence_targets_exclude_missing_labels(self):
        samples = [sample(samples=71), sample(samples=103, index=1)]
        samples[0].target = torch.tensor([0, -100, 1])
        samples[1].target = torch.tensor([1, 0, 1, 0])
        batch = collate(samples)
        self.assertEqual(batch.target.shape, (2, 4))
        self.assertEqual(batch.target[0, -1], -100)
        model = SignalModel(ModelConfig(dim=24, depth=1, task="sequence_classification"))
        self.assertEqual(model(batch)["sequence_logits"].shape, (2, 4, 2))


class ObjectiveTests(unittest.TestCase):
    def test_distillation_dimensions_weights_and_stop_gradient(self):
        student = torch.randn(3, 8, requires_grad=True)
        teachers = [torch.randn(3, 5, requires_grad=True), torch.randn(3, 11, requires_grad=True)]
        module = MultiTeacherLoss(8, [5, 11], weights=[1, 3])
        valid = torch.tensor([[True, True], [False, True], [True, False]])
        loss, weights = module(student, teachers, valid)
        torch.testing.assert_close(weights, torch.tensor([[.25, .75], [0, 1], [1, 0]]))
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertTrue(all(t.grad is None for t in teachers))

    def test_output_semantics_cannot_be_mixed(self):
        with self.assertRaisesRegex(ValueError, "semantics"):
            logit_distillation(torch.ones(2, 2), torch.ones(2, 2), ["left", "right"], ["right", "left"])

    def test_learned_gate_gradient_and_applicability(self):
        student = torch.randn(4, 8, requires_grad=True)
        kd = MultiTeacherLoss(8, [5, 11], learned=True)
        applicable = torch.tensor([[True, False]] * 4)
        loss, weights = kd(student, [torch.randn(4, 5), torch.randn(4, 11)], applicable)
        self.assertTrue(torch.equal(weights[:, 1], torch.zeros(4)))
        loss.backward()
        self.assertIsNotNone(kd.gate.weight.grad)


class DataTests(unittest.TestCase):
    def test_group_leakage_is_rejected(self):
        records = [dict(sample_id=str(i), dataset="d", subject="s", session="0", trial="0",
                        split="train" if i == 0 else "validation") for i in range(2)]
        with self.assertRaisesRegex(ValueError, "leaks"):
            audit_splits(records, "subject")

    def test_normalization_fits_training_only(self):
        s = sample()
        s.x = torch.tensor([[1., 3.]])
        normalizer = Normalizer.fit([s])
        self.assertEqual(normalizer.mean, 2.)
        self.assertEqual(normalizer.std, 1.)

    def test_unknown_channel_identity_rejected(self):
        s = sample()
        s.channel_ids[:] = -1
        s.coord_valid[:] = False
        with self.assertRaisesRegex(ValueError, "genuine"):
            collate([s])

    def test_manifest_hash_and_fewshot_query_preservation(self):
        with tempfile.TemporaryDirectory() as td:
            path = make_synthetic(Path(td) / "data")
            parent = json.loads(path.read_text(encoding="utf-8"))
            selected = Path(td) / "support.json"
            support_manifest(path, selected, 1, seed=17)
            child = json.loads(selected.read_text(encoding="utf-8"))
            self.assertEqual(len([r for r in child["samples"] if r["split"] == "train"]), 2)
            self.assertEqual([r["sample_id"] for r in parent["samples"] if r["split"] != "train"],
                             [r["sample_id"] for r in child["samples"] if r["split"] != "train"])
            file = path.parent / parent["samples"][0]["path"]
            file.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                ManifestDataset(path, "train", 64, "classification")


class RecoveryTests(unittest.TestCase):
    def test_epoch_recovery_matches_uninterrupted(self):
        # Nonzero dropout proves RNG restoration matters, not only saved weights.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = make_synthetic(root / "data")
            config = base_config(objective="ema")
            config["model"]["dropout"] = .1
            config["training"]["epochs"] = 2
            Runner(config, path).fit(root / "continuous")
            short = copy.deepcopy(config)
            short["training"]["epochs"] = 1
            Runner(short, path).fit(root / "resumed")
            Runner(config, path).fit(root / "resumed", resume=root / "resumed" / "last.pt")
            first, second = [load_checkpoint(root / p / "last.pt") for p in ("continuous", "resumed")]
            for key in first["model"]:
                torch.testing.assert_close(first["model"][key], second["model"][key], rtol=0, atol=0)
            for key in first["ema"]:
                torch.testing.assert_close(first["ema"][key], second["ema"][key], rtol=0, atol=0)
            self.assertEqual(first["best"], second["best"])

    def test_resume_rejects_changed_hyperparameters(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = make_synthetic(root / "data")
            config = base_config()
            Runner(config, path).fit(root / "run")
            config["training"]["epochs"] = 2
            config["training"]["lr"] = .02
            with self.assertRaisesRegex(ValueError, "config/data"):
                Runner(config, path).resume(root / "run" / "last.pt")

    def test_resume_preserves_initialization_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = make_synthetic(root / "data")
            config = base_config()
            Runner(config, path).fit(root / "source")
            Runner(config, path).fit(root / "adapted", initialize=root / "source" / "best.pt")
            first = load_checkpoint(root / "adapted" / "last.pt")["initialization"]
            config["training"]["epochs"] = 2
            Runner(config, path).fit(root / "adapted", resume=root / "adapted" / "last.pt")
            state = load_checkpoint(root / "adapted" / "last.pt")
            self.assertEqual(first, state["initialization"])
            environment = json.loads((root / "adapted" / "environment.json").read_text())
            self.assertEqual(first, environment["initialization"])

    def test_test_access_is_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = make_synthetic(root / "data")
            Runner(base_config(), path).fit(root / "run")
            evaluate(root / "run" / "best.pt", path, "test", root / "eval")
            self.assertEqual(len((root / "run" / "test_access.jsonl").read_text().splitlines()), 1)

    def test_evaluation_rejects_repartitioned_training_subjects(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = make_synthetic(root / "data")
            Runner(base_config(), path).fit(root / "run")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            subject = next(r["subject"] for r in manifest["samples"] if r["split"] == "train")
            for r in manifest["samples"]:
                if r["subject"] == subject:
                    r["split"] = "validation"
            new_path = path.parent / "changed.json"
            new_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap checkpoint"):
                evaluate(root / "run" / "best.pt", new_path, "validation", root / "eval")


class DecisionTests(unittest.TestCase):
    def test_unknown_is_pending_and_test_never_selects(self):
        self.assertTrue(all(v["status"] == "pending" for v in decide({}).values()))
        with self.assertRaisesRegex(ValueError, "cannot drive"):
            decide({"S0": {"source": "test"}})

    def test_validation_retain_and_prune(self):
        e = dict(source="validation", real_data=True, frozen_before_runs=True, protocol_id="paired-v1",
                 direction="max", min_gain=.01, max_guard_regression=.02, guards={},
                 paired=[dict(subject=str(i), seed=0, baseline=.5, candidate=.6) for i in range(4)])
        self.assertEqual(decide({"S0": e})["S0"]["status"], "retain")
        for row in e["paired"]:
            row["candidate"] = .49
        self.assertEqual(decide({"S0": e})["S0"]["status"], "prune")

    def test_known_metric(self):
        result = metrics([[5, 0], [5, 0], [0, 5], [5, 0]], [0, 0, 1, 1], "classification")
        self.assertEqual(result["accuracy"], .75)
        self.assertEqual(result["balanced_accuracy"], .75)

    def test_guard_regression_overrides_main_gain(self):
        e = dict(source="validation", real_data=True, frozen_before_runs=True, protocol_id="paired-v1",
                 direction="max", min_gain=.01, max_guard_regression=.02,
                 guards={"critical": dict(baseline=.7, candidate=.6, direction="max")},
                 paired=[dict(subject=str(i), seed=0, baseline=.5, candidate=.6) for i in range(4)])
        self.assertEqual(decide({"S0": e})["S0"]["status"], "prune")


if __name__ == "__main__":
    unittest.main()
