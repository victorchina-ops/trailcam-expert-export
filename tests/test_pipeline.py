"""Integration checks for incremental exports without model downloads or GPUs."""
from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from trailcam import __main__ as entry
from trailcam.attributes import AttributeEngine
from trailcam.contacts import features, make_contact_sheet, select_examples
from trailcam.export import FIELDS, make_row
from trailcam.fusion import COUNT_FIELDS, fuse
from trailcam.storage import Cache, configuration, file_hash, write_csv


def fake_result():
    def counts(people, limited=False):
        values = {field: 0 for field in COUNT_FIELDS}
        values["people_total"] = people
        if limited:
            for field in ("strollers", "atv_utv", "other_vehicles", "kick_scooters"):
                values[field] = None
        return values

    people = [{"person_id": f"person_{index:04d}", "bbox_xyxy": [20, 20, 120, 200],
               "pose": {"orientation": "side", "facing_direction": "left"},
               "mask_backpack": None} for index in (1, 2)]
    yoloe = counts(2)
    yoloe["backpacks"] = 1
    md = dict.fromkeys(COUNT_FIELDS)
    md["people_total"] = 3
    return {
        "width": 200, "height": 300, "device": "cpu", "persons": people,
        "experts": {
            "yoloe": {"counts": yoloe, "detections": []},
            "yolo26n": {"counts": counts(3, True), "detections": []},
            "megadetector": {"counts": md, "detections": []},
            "pose": {"counts": {"people_total": 2}, "detections": [],
                     "direction_counts": {"left": 2, "right": 0, "toward": 0, "away": 0, "unclear": 0},
                     "orientation_counts": {"front": 0, "back": 0, "side": 2, "unknown": 0}},
        }, "timings": {"vision_total_seconds": 0.001},
    }


def fake_attributes(persons, failed=False):
    results = []
    for index, person in enumerate(persons):
        error = failed and index == 0
        attributes = {
            "status": "error" if error else "ok", "error": "failed crop" if error else None,
            "native_age_label": "unknown" if error else "18_60",
            "native_orientation_label": "unknown" if error else "side",
            "backpack_presence": None if error else False,
            "presentation_proxy": "unclear", "scores": {}, "device": "cpu",
        }
        results.append({**person, "attributes": attributes})
    return {"device": "cpu", "seconds": 0.001, "persons": results}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.output = self.root / "exports"
        self.cache = self.root / "cache"
        self.models = self.root / "models"
        self.models.mkdir()
        self.vision_calls = []
        self.attribute_calls = []
        self.vision = Mock()
        self.vision.analyze.side_effect = self._vision_analyze
        self.attribute = Mock()
        self.attribute.analyze.side_effect = self._attribute_analyze
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.ensure = self.stack.enter_context(patch("trailcam.model_setup.ensure_models"))
        self.vision_constructor = self.stack.enter_context(patch("trailcam.vision.VisionEngine", return_value=self.vision))
        self.attribute_constructor = self.stack.enter_context(patch("trailcam.attributes.AttributeEngine", return_value=self.attribute))
        self.stack.enter_context(patch.object(entry, "configuration", return_value=("test_config", {"test": True})))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def _vision_analyze(self, path):
        self.vision_calls.append(Path(path).name)
        with Image.open(path) as image:
            image.verify()
        return fake_result()

    def _attribute_analyze(self, path, persons):
        self.attribute_calls.append(Path(path).name)
        return fake_attributes(persons)

    def image(self, name, color="red"):
        path = self.images / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (200, 300), color).save(path)
        return path

    def args(self, *extra):
        return entry.arguments(["--input", str(self.images), "--output", str(self.output),
                                "--models", str(self.models), "--cache", str(self.cache),
                                "--no-contact-sheet", *extra])

    def last_export(self):
        summary_file = sorted(self.output.glob("export_*/export_*_run.json"))[-1]
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        with (summary_file.parent / summary["csv"]).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        return summary, rows

    def test_second_run_uses_cache_and_new_content_reanalyzes(self):
        self.image("one.jpg")
        self.image("nested/two.png", "blue")
        args = self.args()
        self.assertEqual(entry.run(args), 0)
        first, first_rows = self.last_export()
        self.assertEqual((first["analyzed_images"], first["cached_images"]), (2, 0))
        self.assertEqual(entry.run(args), 0)
        second, second_rows = self.last_export()
        self.assertEqual((second["analyzed_images"], second["cached_images"]), (0, 2))
        self.assertEqual(len(self.vision_calls), 2)
        self.assertEqual(self.vision_constructor.call_count, 1)
        self.assertEqual(len(second_rows), 2)
        self.assertNotEqual(first["csv"], second["csv"])
        first_folder = self.output / first["export_subfolder"]
        second_folder = self.output / second["export_subfolder"]
        self.assertNotEqual(first_folder, second_folder)
        self.assertEqual(first_folder.name, "export_" + first["run_id"])
        self.assertEqual(second_folder.name, "export_" + second["run_id"])
        self.assertTrue((first_folder / first["csv"]).is_file())
        self.assertTrue((second_folder / second["csv"]).is_file())
        self.assertEqual(len(list(first_folder.iterdir())), 2)
        self.assertEqual(len(list(second_folder.iterdir())), 2)
        self.assertTrue(all(path.is_dir() for path in self.output.iterdir()))
        self.assertEqual({row["sha256"] for row in first_rows}, {row["sha256"] for row in second_rows})
        self.image("one.jpg", "green")
        self.assertEqual(entry.run(args), 0)
        third, _ = self.last_export()
        self.assertEqual((third["analyzed_images"], third["cached_images"]), (1, 1))
        self.assertEqual(len(self.vision_calls), 3)

    def test_corrupt_image_is_exported_as_error_and_retried(self):
        self.image("ok.jpg")
        (self.images / "broken.jpg").write_bytes(b"not an image")
        args = self.args()
        self.assertEqual(entry.run(args), 1)
        summary, rows = self.last_export()
        error = next(row for row in rows if row["relative_path"] == "broken.jpg")
        self.assertEqual(error["status"], "error")
        self.assertTrue(error["error"])
        self.assertEqual(error["combined_people_total"], "")
        self.assertEqual(summary["images_with_errors"], 1)
        self.assertEqual(entry.run(args), 1)
        self.assertEqual(self.vision_calls.count("broken.jpg"), 2)
        self.assertEqual(self.vision_calls.count("ok.jpg"), 1)
        self.image("broken.jpg", "blue")
        self.assertEqual(entry.run(args), 0)
        self.assertEqual(self.last_export()[0]["images_with_errors"], 0)

    def test_partial_attribute_errors_preserve_counts_and_are_retried(self):
        self.image("one.jpg")
        self.attribute.analyze.side_effect = lambda path, persons: fake_attributes(persons, failed=True)
        self.assertEqual(entry.run(self.args()), 1)
        _, rows = self.last_export()
        self.assertEqual(rows[0]["status"], "partial_error")
        self.assertEqual(rows[0]["combined_people_total"], "2")
        self.assertIn("failed crop", rows[0]["error"])
        self.assertEqual(entry.run(self.args()), 1)
        self.assertEqual(len(self.vision_calls), 2)

    def test_no_contact_sheet_flag_skips_renderer(self):
        self.image("one.jpg")
        with patch("trailcam.contacts.make_contact_sheet") as render:
            self.assertEqual(entry.run(self.args("--no-contact-sheet")), 0)
            render.assert_not_called()
        self.assertIsNone(self.last_export()[0]["contact_sheet"])

    def test_default_enables_twenty_image_contact_sheet(self):
        with patch.object(entry, "ROOT", self.root):
            args = entry.arguments([])
        self.assertTrue(args.contact_sheet)
        self.assertEqual(args.contact_sheet_size, 20)
        self.image("one.jpg")
        self.assertEqual(entry.run(self.args("--contact-sheet")), 0)
        summary, _ = self.last_export()
        self.assertEqual(len(summary["contact_sheet_selection"]), 1)
        folder = self.output / summary["export_subfolder"]
        self.assertEqual({p.name for p in folder.iterdir()}, {
            summary["csv"], summary["contact_sheet"], "export_" + summary["run_id"] + "_run.json"})
        with Image.open(self.output / summary["export_subfolder"] / summary["contact_sheet"]) as image:
            self.assertEqual(image.format, "JPEG")

    def test_csv_retains_canonical_totals_and_expert_disagreements(self):
        self.image("one.jpg")
        self.assertEqual(entry.run(self.args()), 0)
        _, rows = self.last_export()
        row = rows[0]
        self.assertEqual(set(row), set(FIELDS))
        self.assertEqual(row["combined_people_total"], "2")
        self.assertEqual(row["vote_people_total"], "3")
        self.assertEqual(row["combined_age_unknown"], "2")
        self.assertEqual(row["yolo26n_strollers"], "")
        self.assertEqual(row["yoloe_strollers"], "0")
        self.assertEqual(row["megadetector_dogs"], "")
        self.assertEqual(row["combined_dogs"], "0")
        self.assertEqual(sum(int(row["combined_direction_" + direction]) for direction in
                             ("left", "right", "toward", "away", "unclear")), 2)
        self.assertEqual(len(json.loads(row["persons_json"])), 2)
        self.assertEqual(set(json.loads(row["expert_evidence_json"])), {"yoloe", "yolo26n", "megadetector", "pose"})
        self.assertEqual(row["attribute_device"], "cpu")

    def test_new_only_and_force_options(self):
        self.image("one.jpg")
        self.assertEqual(entry.run(self.args()), 0)
        self.assertEqual(entry.run(self.args("--export-new-only")), 0)
        self.assertEqual(self.last_export()[1], [])
        self.assertEqual(entry.run(self.args("--force", "--export-new-only")), 0)
        self.assertEqual(len(self.last_export()[1]), 1)
        self.assertEqual(len(self.vision_calls), 2)

    def test_corrupt_cache_structure_is_recomputed(self):
        path = self.image("one.jpg")
        self.assertEqual(entry.run(self.args()), 0)
        cache_file = self.cache / "test_config" / (file_hash(path) + ".json")
        valid = json.loads(cache_file.read_text(encoding="utf-8"))
        bad_person = copy.deepcopy(valid)
        bad_person["persons"] = [{}]
        wrong_total = copy.deepcopy(valid)
        wrong_total["combined"]["counts"]["people_total"] = 99
        for malformed in (["unexpected root array"],
                          {"status": "ok", "sha256": file_hash(path), "_cache_format": 1},
                          bad_person, wrong_total):
            with self.subTest(malformed=malformed):
                cache_file.write_text(json.dumps(malformed), encoding="utf-8")
                self.assertEqual(entry.run(self.args()), 0)
                summary, rows = self.last_export()
                self.assertEqual(summary["analyzed_images"], 1)
                self.assertEqual(rows[0]["status"], "ok")

    def test_active_lock_blocks_model_setup(self):
        self.image("one.jpg")
        self.cache.mkdir()
        lock = self.cache / "run.lock"
        lock.write_text("another process")
        with self.assertRaises(RuntimeError):
            entry.run(self.args())
        self.ensure.assert_not_called()
        self.assertEqual(lock.read_text(), "another process")

    def test_recursive_discovery_excludes_output_models_and_hidden(self):
        self.image("root.jpg")
        self.image("child/child.jpg")
        self.image(".hidden/secret.jpg")
        nested_output = self.images / "exports"
        nested_output.mkdir()
        Image.new("RGB", (20, 20)).save(nested_output / "old_contactsheet.jpg")
        discovered = entry.discover(self.images, excluded=(nested_output,))
        self.assertEqual({path.relative_to(self.images).as_posix() for path in discovered},
                         {"root.jpg", "child/child.jpg"})
        self.assertEqual([path.name for path in entry.discover(self.images, recursive=False)], ["root.jpg"])


class DiversityAndFusionTests(unittest.TestCase):
    def items(self):
        items = []
        for index in range(35):
            result = fake_result()
            result["status"] = "ok"
            result["persons"] = fake_attributes(result["persons"])["persons"]
            if index == 29:
                result["experts"]["yoloe"]["counts"]["dogs"] = 1
                result["experts"]["yolo26n"]["counts"]["dogs"] = 1
            if index == 32:
                result["experts"]["yoloe"]["counts"]["strollers"] = 1
            if index == 34:
                result["experts"]["yoloe"]["counts"]["atv_utv"] = 1
            fuse(result)
            items.append({"relative_path": f"frame_{index:03d}.jpg", "cache_hit": False,
                          "result": result, "image_id": f"img{index}"})
        return items

    def test_contact_selection_is_twenty_diverse_unique_and_reproducible(self):
        items = self.items()
        first = select_examples(items, 20)
        second = select_examples(list(reversed(items)), 20)
        self.assertEqual(len(first), 20)
        self.assertEqual(len({item["image_id"] for item in first}), 20)
        self.assertEqual([item["image_id"] for item in first], [item["image_id"] for item in second])
        covered = set().union(*(features(item) for item in first))
        self.assertTrue({"dogs", "strollers", "atv_utv", "backpacks"}.issubset(covered))

    def test_orientation_conflict_and_mask_conflict_abstain(self):
        result = fake_result()
        result["persons"] = fake_attributes(result["persons"])["persons"]
        person = result["persons"][0]
        person["attributes"]["native_orientation_label"] = "back"
        person["mask_backpack"] = True
        fuse(result)
        self.assertEqual(person["combined"]["orientation"], "unknown")
        self.assertEqual(person["combined"]["direction"], "unclear")
        self.assertIsNone(person["combined"]["carrying_backpack"])
        self.assertEqual(person["combined"]["backpack_source"], "attribute_mask_conflict")

    def test_model_change_invalidates_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            model = models / "test.pt"
            model.write_bytes(b"first weights")
            first, _ = configuration(models, "auto", 8)
            model.write_bytes(b"changed weights")
            second, _ = configuration(models, "auto", 8)
        self.assertNotEqual(first, second)

    def test_csv_filename_formula_is_escaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "export.csv"
            write_csv(path, [{"relative_path": "=malicious.jpg", "count": 0}], ["relative_path", "count"])
            with path.open(encoding="utf-8-sig", newline="") as source:
                row = next(csv.DictReader(source))
        self.assertEqual(row["relative_path"], "'=malicious.jpg")
        self.assertEqual(row["count"], "0")

    def test_attribute_cuda_failure_records_cpu_after_fallback(self):
        import numpy as np
        engine = object.__new__(AttributeEngine)
        engine.device = "cuda:0"
        engine.requested_device = "auto"
        engine.fallback_reasons = []
        engine._np = np
        import cv2
        engine._cv2 = cv2
        engine.predictor = object()
        calls = []

        def run(tensor):
            calls.append(engine.device)
            if engine.device.startswith("cuda"):
                raise RuntimeError("CUDA out of memory")
            return np.zeros(26, dtype=np.float32)

        def cpu():
            engine.device = "cpu"

        engine._run_tensor = run
        engine._init_cpu = cpu
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "person.png"
            Image.new("RGB", (100, 200)).save(image_path)
            result = engine.analyze(image_path, [{"person_id": "person_0001", "bbox_xyxy": [0, 0, 100, 200]}])
        self.assertEqual(calls, ["cuda:0", "cpu"])
        self.assertEqual(engine.device, "cpu")
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["persons"][0]["attributes"]["device"], "cpu")
        self.assertEqual(result["persons"][0]["attributes"]["scores"]["native_female"], 0.0)
        self.assertIn("CUDA inference failed", engine.fallback_reasons[0])


if __name__ == "__main__":
    unittest.main()
