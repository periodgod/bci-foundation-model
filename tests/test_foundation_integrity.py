"""Scientific invariants for real foundation studies; no score thresholds."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from eeg_mime_foundation.data import make_synthetic, sha256
from eeg_mime_foundation.engine import Runner, evaluate, load_checkpoint, save_checkpoint, training_class_weights
from eeg_mime_foundation.smoke import base_config
from eeg_mime_foundation.realdata import source_spec


class FoundationIntegrityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_class_weights_use_training_targets(self):
        class Item:
            def __init__(self, target):
                self.target = torch.tensor(target)
        weights, counts = training_class_weights([Item([0, 0, -100]), Item([1])], 2, "cpu")
        self.assertEqual(counts, [2, 1])
        torch.testing.assert_close(weights, torch.tensor([.75, 1.5]))
        with self.assertRaisesRegex(ValueError, "all classes"):
            training_class_weights([Item([0])], 2, "cpu")

    def test_weighted_loss_and_gradient(self):
        with tempfile.TemporaryDirectory() as td:
            path = make_synthetic(Path(td) / "data")
            config = base_config()
            config["training"]["class_balance"] = "inverse_frequency"
            runner = Runner(config, path)
            runner.class_weights = torch.tensor([1., 3.])
            logits = torch.tensor([[2., 0.], [2., 0.]], requires_grad=True)
            labels = torch.tensor([0, 1])
            loss = runner.supervised_loss({"logits": logits}, labels)
            torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(logits, labels, weight=runner.class_weights))
            loss.backward()
            self.assertGreater(float(logits.grad[1].norm()), float(logits.grad[0].norm()))

    def test_initialization_and_teacher_reject_transitive_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = make_synthetic(root / "data")
            config = base_config()
            runner = Runner(config, path)
            state = runner.checkpoint(0)
            record = runner.data["validation"].records[0]
            state["training_groups"].append([record["dataset"], record["subject"]])
            checkpoint = root / "contaminated.pt"
            save_checkpoint(checkpoint, state)
            with self.assertRaisesRegex(ValueError, "overlap held-out"):
                Runner(config, path).initialize(checkpoint)
            distill = copy.deepcopy(config)
            distill["training"]["objective"] = "distill"
            distill["teachers"] = [dict(checkpoint=str(checkpoint), sha256=sha256(checkpoint),
                                       use_status="approved", provenance="local scratch", overlap="audited")]
            with self.assertRaisesRegex(ValueError, "overlap held-out"):
                Runner(distill, path)

    def test_transitive_groups_survive_initialization(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = make_synthetic(root / "data")
            runner = Runner(base_config(), path)
            state = runner.checkpoint(0)
            state["training_groups"].append(["synthetic", "additional_train_subject"])
            state["training_signal_hashes"].append("additional_train_waveform")
            checkpoint = root / "source.pt"
            save_checkpoint(checkpoint, state)
            runner.initialize(checkpoint)
            derived = runner.checkpoint(1)
            self.assertIn(["synthetic", "additional_train_subject"], derived["training_groups"])
            self.assertIn("additional_train_waveform", derived["training_signal_hashes"])

    def test_source_selection_cannot_touch_reserved_test(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = make_synthetic(root / "data")
            runner = Runner(base_config(), path)
            state = runner.checkpoint(0)
            test = next(r for r in runner.data["train"].manifest["samples"] if r["split"] == "test")
            state["selection_groups"].append([test["dataset"], test["subject"]])
            checkpoint = root / "selected_on_test.pt"
            save_checkpoint(checkpoint, state)
            with self.assertRaisesRegex(ValueError, "selection groups"):
                runner.initialize(checkpoint)

    def test_source_waveform_overlap_is_rejected_without_matching_group_names(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = make_synthetic(root / "data")
            runner = Runner(base_config(), path)
            state = runner.checkpoint(0)
            state["training_signal_hashes"].append(next(h for h,split in runner.data["train"].signal_hashes.items() if split=="validation"))
            checkpoint = root/"waveform_overlap.pt"
            save_checkpoint(checkpoint,state)
            with self.assertRaisesRegex(ValueError, "waveforms overlap"):
                runner.initialize(checkpoint)

    def test_source_grouping_must_be_comparable(self):
        with tempfile.TemporaryDirectory() as td:
            path = make_synthetic(Path(td)/"data")
            runner = Runner(base_config(),path)
            state = runner.checkpoint(0)
            state["group_by"] = "session"
            with self.assertRaisesRegex(ValueError,"grouping differs"):
                runner.import_provenance(state)

    def test_development_cohort_preserves_reserved_subjects(self):
        for kind in ("mi","sleep"):
            original = source_spec(kind,"smoke")["subjects"]
            expanded = source_spec(kind,"development")["subjects"]
            self.assertEqual(len(expanded),10)
            self.assertEqual({s for s,split in original.items() if split=="test"},
                             {s for s,split in expanded.items() if split=="test"})
            self.assertTrue(all(expanded[s]==split for s,split in original.items()))
            self.assertEqual([sum(split==x for split in expanded.values()) for x in ("train","validation","test")],[6,3,1])

    def test_distillation_can_select_subject_accuracy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = make_synthetic(root/"data")
            config = base_config()
            teacher = root/"teacher.pt"
            save_checkpoint(teacher,Runner(config,path).checkpoint(0))
            config["training"].update(objective="distill",selection_metric="balanced_accuracy",direction="max")
            config["teachers"] = [dict(checkpoint=str(teacher),sha256=sha256(teacher),
                use_status="approved",provenance="local synthetic engineering",overlap="enforced")]
            config["distillation"] = dict(supervised_weight=1.,logit_weight=.1,learned=False)
            runner = Runner(config,path)
            self.assertIn("balanced_accuracy",runner.validation()["subject_macro"])
            runner.fit(root/"student")
            self.assertTrue((root/"student"/"best.pt").is_file())

    def test_evaluation_grouping_cannot_silently_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = make_synthetic(root/"data")
            checkpoint = root/"model.pt"
            save_checkpoint(checkpoint,Runner(base_config(),path).checkpoint(0))
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["group_by"] = "trial"
            changed = path.parent/"grouping_changed.json"
            changed.write_text(json.dumps(manifest),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"grouping differs"):
                evaluate(checkpoint,changed,"validation",root/"evaluation")

    def test_dataset_batches_keep_shapes_separate_and_cover_training(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = make_synthetic(root/"data")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            for record in manifest["samples"]:
                record["dataset"] = f"dataset_{int(record['subject'])%2}"
            path.write_text(json.dumps(manifest),encoding="utf-8")
            config = base_config()
            config["training"]["batch_by_dataset"] = True
            runner = Runner(config,path)
            seen = []
            for batch in runner.loader("train",True):
                self.assertEqual(len({r["dataset"] for r in batch.metadata}),1)
                seen.extend(r["sample_id"] for r in batch.metadata)
            self.assertEqual(sorted(seen),sorted(r["sample_id"] for r in runner.data["train"].records))
            self.assertEqual(len(seen),len(set(seen)))

    def test_dataset_bucket_recovery_is_exact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = make_synthetic(root/"data")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            for record in manifest["samples"]:
                record["dataset"] = f"dataset_{int(record['subject'])%2}"
            path.write_text(json.dumps(manifest),encoding="utf-8")
            config = base_config(objective="ema")
            config["model"]["dropout"] = .1
            config["training"].update(batch_by_dataset=True,epochs=2)
            Runner(config,path).fit(root/"continuous")
            short = copy.deepcopy(config)
            short["training"]["epochs"] = 1
            Runner(short,path).fit(root/"recovered")
            Runner(config,path).fit(root/"recovered",resume=root/"recovered"/"last.pt")
            first,second = [load_checkpoint(root/name/"last.pt") for name in ("continuous","recovered")]
            for component in ("model","ema"):
                for name in first[component]:
                    torch.testing.assert_close(first[component][name],second[component][name],rtol=0,atol=0)


if __name__ == "__main__":
    unittest.main()
