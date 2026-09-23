"""Integration tests of the v2 export pipeline (``python -m trailcam``) without models or GPUs.

``VisionEngine``, ``AttributeEngine`` and ``ensure_models`` are replaced by the
scene-driven fakes in ``fakes_v2``; everything else is real: discovery, file
hashing, the inference cache (format 2), per-image and cross-image
post-processing (roster acceptance, object verification, age geometry,
events, direction), the images / events / camera-days CSVs, the run JSON and
the contact sheet.
"""
from __future__ import annotations

import contextlib
import copy
import csv
from datetime import datetime
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

try:
    from . import fakes_v2 as fakes
except ImportError:  # unittest discover -s tests imports test modules top-level
    import fakes_v2 as fakes

from trailcam import __main__ as entry
from trailcam import __version__, roster
from trailcam.attributes import AttributeEngine
from trailcam.contacts import features, select_examples
from trailcam.export import EVENT_FIELDS, FIELDS, SUMMARY_FIELDS
from trailcam.fusion import fuse
from trailcam.postprocess import postprocess
from trailcam.storage import CACHE_FORMAT, configuration, file_hash, write_csv
from trailcam.vision import PROFILES, VisionEngine

EXCEL_CELL_LIMIT = 32767
WALK = ("camA/IMG_0001.jpg", "camA/IMG_0002.jpg", "camA/IMG_0003.jpg")
GROUP = "camA/IMG_0010.jpg"
EMPTY = "camB/IMG_0100.jpg"
TRAIL = WALK + (GROUP, EMPTY)
CHILD_COLOR = (40, 160, 200)
RUN_SPECIFIC = ("run_id", "cache_hit", "current_run_seconds")
RUN_JSON_KEYS = {
    "run_id", "version", "input_dir", "export_subfolder", "csv", "events_csv", "camera_days_csv",
    "event_count", "camera_calibrations", "postprocess_settings", "images_with_postprocess_errors",
    "image_count", "csv_rows", "cached_images", "analyzed_images", "images_with_errors",
    "model_load_seconds", "total_seconds", "configuration_hash", "configuration",
    "contact_sheet", "contact_sheet_error", "contact_sheet_selection",
    "age_model", "age_mode", "age_model_failures"}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames


def without(row, keys=RUN_SPECIFIC):
    return {key: value for key, value in row.items() if key not in keys}


class Export:
    """One export subfolder: run JSON plus the three CSVs."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.summary = json.loads((self.folder / (self.folder.name + "_run.json")).read_text(encoding="utf-8"))
        self.rows, self.header = read_csv(self.folder / self.summary["csv"])
        self.events, self.event_header = (read_csv(self.folder / self.summary["events_csv"])
                                          if self.summary["events_csv"] else ([], []))
        self.days, self.day_header = (read_csv(self.folder / self.summary["camera_days_csv"])
                                      if self.summary["camera_days_csv"] else ([], []))

    def row(self, relative_path):
        matches = [row for row in self.rows if row["relative_path"] == relative_path]
        assert len(matches) == 1, (relative_path, [row["relative_path"] for row in self.rows])
        return matches[0]

    def event_of(self, relative_path):
        event_id = self.row(relative_path)["event_id"]
        return next(event for event in self.events if event["event_id"] == event_id)

    def day(self, camera_id, date):
        return next(row for row in self.days if (row["camera_id"], row["date"]) == (camera_id, date))


class PipelineCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.images = self.root / "images"
        self.images.mkdir()
        self.output = self.root / "exports"
        self.cache = self.root / "cache"
        self.models = self.root / "models"
        self.models.mkdir()
        self.scenes = {}
        self.vision = fakes.FakeVisionEngine(self.scenes, root=self.images)
        self.attributes = fakes.FakeAttributeEngine()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.ensure = stack.enter_context(patch("trailcam.model_setup.ensure_models"))
        self.vision_constructor = stack.enter_context(
            patch("trailcam.vision.VisionEngine", side_effect=self._make_vision))
        self.attribute_constructor = stack.enter_context(
            patch("trailcam.attributes.AttributeEngine", return_value=self.attributes))
        # A settings.json next to the real package must not change test defaults.
        stack.enter_context(patch.object(entry, "ROOT", self.root))
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        stack.enter_context(contextlib.redirect_stdout(self.stdout))
        stack.enter_context(contextlib.redirect_stderr(self.stderr))

    def _make_vision(self, models_dir, device="auto", threads=8, profile="standard", empty_frame_gate=True):
        self.vision.profile, self.vision.empty_frame_gate = profile, empty_frame_gate
        return self.vision

    def add(self, relative, spec=None, when=None, **exif):
        """Write an image rendering ``spec``; ``when`` is an EXIF 'YYYY:MM:DD HH:MM:SS' time."""
        spec = fakes.scene() if spec is None else spec
        self.scenes[relative] = spec
        return fakes.write_image(self.images / relative, spec, when, **exif)

    def trail(self):
        """Two cameras: a person walking right over three frames, a later group
        (adult, child, dog, one faint rejected candidate), and an empty frame
        whose only detection lies in the camera data strip."""
        for frame, (relative, x) in enumerate(zip(WALK, (60, 200, 340))):
            self.add(relative, fakes.scene(fakes.walker(x)), f"2026:05:01 10:00:{10 * frame:02d}")
        self.add(GROUP, fakes.scene(
            fakes.walker(100), fakes.walker(260, y=170, height=130, color=CHILD_COLOR),
            fakes.thing("dog", [450, 250, 560, 320], {"yoloe": .8, "yolo26n": .7, "megadetector": .6}, (120, 80, 40)),
            fakes.person([520, 60, 560, 160], {"yolo26n": .15}, color=(150, 150, 150))), "2026:05:01 12:00:00")
        self.add(EMPTY, fakes.scene(fakes.person([300, 460, 380, 478], {"yolo26n": .9, "megadetector": .5}),
                                    strip_rows=24), "2026:05:02 08:00:00")

    def args(self, *extra):
        return entry.arguments(["--input", str(self.images), "--output", str(self.output),
                                "--models", str(self.models), "--cache", str(self.cache),
                                "--no-contact-sheet", "--events", "--geometry-age", "--large-bags",
                                "--near-fraction", "0.07", "--empty-frame-gate", *extra])

    def run_export(self, *extra, expect=0):
        before = set(self.output.glob("export_*")) if self.output.exists() else set()
        code = entry.run(self.args(*extra))
        self.assertEqual(code, expect, self.stderr.getvalue())
        created = sorted(set(self.output.glob("export_*")) - before)
        self.assertEqual(len(created), 1, created)
        return Export(created[0])

    def cache_path(self, export, relative):
        return self.cache / export.summary["configuration_hash"] / (file_hash(self.images / relative) + ".json")


class FirstRunTests(PipelineCase):
    def test_first_run_analyzes_every_image_and_caches_raw_inference(self):
        self.trail()
        export = self.run_export()
        summary = export.summary
        self.assertEqual((summary["image_count"], summary["analyzed_images"], summary["cached_images"],
                          summary["images_with_errors"], summary["csv_rows"]), (5, 5, 0, 0, 5))
        self.assertEqual(sorted(self.vision.calls), sorted(TRAIL))
        self.assertEqual(sorted(call["name"] for call in self.attributes.calls), sorted(Path(p).name for p in TRAIL))
        self.ensure.assert_called_once_with(self.models, offline=False)
        self.vision_constructor.assert_called_once_with(self.models, "auto", 8, "standard", True)
        self.attribute_constructor.assert_called_once_with(self.models, "auto", 8)
        folder = self.cache / summary["configuration_hash"]
        self.assertEqual({path.stem for path in folder.glob("*.json")}, {row["sha256"] for row in export.rows})
        for relative in TRAIL:
            with self.subTest(relative=relative):
                row = export.row(relative)
                self.assertEqual(row["sha256"], file_hash(self.images / relative))
                self.assertEqual(row["image_id"], "img_" + row["sha256"][:16])
                cached = json.loads(self.cache_path(export, relative).read_text(encoding="utf-8"))
                self.assertEqual(cached["_cache_format"], CACHE_FORMAT)
                self.assertEqual(CACHE_FORMAT, 2)
                self.assertEqual((cached["sha256"], cached["status"]), (row["sha256"], "ok"))
                self.assertNotIn("_decoded_image", cached)
                self.assertEqual(set(cached["experts"]), {"yoloe", "yolo26n", "megadetector", "pose"})
                # Only inference is cached; post-processing is recomputed on every run.
                for key in ("combined", "votes", "objects", "post", "postprocess_error"):
                    self.assertNotIn(key, cached)
                for person in cached["persons"]:
                    self.assertNotIn("accepted", person)
                    self.assertEqual(person["attributes"]["status"], "ok")
                    self.assertEqual(person["appearance"]["version"], "hsv_v1")
                self.assertEqual(cached["attribute_device"], "cpu")
                self.assertNotIn("persons", cached["attribute_runtime"])
                self.assertIn("attributes_seconds", cached["timings"])
        group = json.loads(self.cache_path(export, GROUP).read_text(encoding="utf-8"))
        # Raw detections are kept below the acceptance thresholds (tuning needs no re-inference).
        self.assertIn(0.15, [d["confidence"] for d in group["experts"]["yolo26n"]["detections"]])
        self.assertEqual(len(group["persons"]), 3)
        gated = json.loads(self.cache_path(export, EMPTY).read_text(encoding="utf-8"))
        self.assertTrue(gated["gated_empty"])
        self.assertEqual(gated["experts"]["yoloe"]["skipped"], "empty_frame_gate")

    def test_images_csv_counts_appearances_in_each_photo(self):
        self.trail()
        export = self.run_export()
        self.assertEqual(export.header, FIELDS)
        self.assertEqual([row["relative_path"] for row in export.rows], sorted(TRAIL, key=str.casefold))
        walk_event = {export.row(relative)["event_id"] for relative in WALK}
        self.assertEqual(len(walk_event), 1)
        for frame, relative in enumerate(WALK, 1):
            with self.subTest(relative=relative):
                row = export.row(relative)
                self.assertEqual((row["status"], row["error"], row["cache_hit"]), ("ok", "", "False"))
                self.assertEqual((row["camera_id"], row["event_frame"], row["event_image_count"]), ("camA", str(frame), "3"))
                self.assertEqual((row["capture_time"], row["time_source"], row["sequence_number"]),
                                 (f"2026-05-01T10:00:{10 * (frame - 1):02d}", "exif_original", str(frame)))
                self.assertEqual((row["people_total"], row["dir_right"], row["direction_from_motion"]), ("1", "1", "1"))
                self.assertEqual((row["adults"], row["children"], row["age_unknown"]), ("0", "0", "1"))
                self.assertEqual(row["camera_calibration_status"], "insufficient")
                person, = json.loads(row["persons_json"])
                self.assertEqual((person["direction"], person["direction_source"]), ("right", "motion"))
        group = export.row(GROUP)
        self.assertEqual((group["people_total"], group["adults"], group["children"], group["age_unknown"]),
                         ("2", "0", "1", "1"))
        self.assertEqual((group["age_by_relative_height"], group["age_by_camera_calibration"]), ("1", "0"))
        self.assertEqual((group["candidates_total"], group["candidates_rejected"]), ("3", "1"))
        self.assertEqual((group["dir_toward"], group["direction_from_facing"], group["direction_from_motion"]),
                         ("2", "2", "0"))
        self.assertEqual(group["dogs"], "1")
        self.assertEqual({p["age"] for p in json.loads(group["persons_json"])}, {"unknown", "child"})
        # Blank = the expert cannot report the category; 0 = assessed and absent.
        self.assertEqual((group["yolo26n_strollers"], group["megadetector_dogs"]), ("", ""))
        self.assertEqual((group["yoloe_strollers"], group["strollers"], group["bicycles"]), ("0", "0", "0"))
        self.assertEqual(set(json.loads(group["expert_evidence_json"])), {"yoloe", "yolo26n", "megadetector", "pose"})
        empty = export.row(EMPTY)
        self.assertEqual((empty["camera_id"], empty["gated_empty"], empty["people_total"]), ("camB", "True", "0"))
        self.assertEqual((empty["adults"], empty["children"], empty["age_unknown"], empty["dogs"]), ("0", "0", "0", "0"))
        self.assertGreater(int(empty["data_strip_bottom"]), 0)
        self.assertEqual(empty["persons_json"], "[]")
        self.assertEqual((empty["candidates_total"], empty["candidates_rejected"]), ("0", "0"))
        # v2: experts skipped by the gate report blank counts; the cheap experts ran and saw nothing countable.
        self.assertEqual((empty["yoloe_people_total"], empty["yoloe_strollers"], empty["yoloe_dogs"]), ("", "", ""))
        self.assertEqual((empty["yolo26n_people_total"], empty["megadetector_people_total"]), ("0", "0"))
        # v2: a camera without people has an "insufficient" (not missing) calibration.
        self.assertEqual(empty["camera_calibration_status"], "insufficient")
        # Five appearances in the photos; the events CSV holds three distinct people.
        self.assertEqual(sum(int(row["people_total"]) for row in export.rows), 5)

    def test_events_csv_deduplicates_people_across_frames(self):
        self.trail()
        export = self.run_export()
        self.assertEqual(export.event_header, EVENT_FIELDS)
        self.assertEqual(len(export.events), 3)
        self.assertEqual({event["event_id"] for event in export.events}, {row["event_id"] for row in export.rows})
        walk = export.event_of(WALK[0])
        self.assertEqual((walk["camera_id"], walk["image_count"], walk["people_unique"], walk["people_max_frame"]),
                         ("camA", "3", "1", "1"))
        self.assertEqual((walk["start"], walk["end"], float(walk["duration_seconds"])),
                         ("2026-05-01T10:00:00", "2026-05-01T10:00:20", 20.0))
        self.assertEqual((walk["dir_right"], walk["direction_from_motion"], walk["age_unknown"]), ("1", "1", "1"))
        self.assertEqual(json.loads(walk["images"]), list(WALK))  # v2: relative paths, not image ids
        self.assertEqual(walk["needs_review"], "False")
        group = export.event_of(GROUP)
        self.assertEqual((group["people_unique"], group["adults"], group["children"], group["dogs"]),
                         ("2", "0", "1", "1"))
        self.assertEqual((group["dir_toward"], group["direction_from_facing"]), ("2", "2"))
        empty = export.event_of(EMPTY)
        self.assertEqual((empty["camera_id"], empty["people_unique"], empty["image_count"]), ("camB", "0", "1"))

    def test_camera_days_csv_sums_event_totals(self):
        self.trail()
        export = self.run_export()
        self.assertEqual(export.day_header, SUMMARY_FIELDS)
        self.assertEqual([(row["camera_id"], row["date"]) for row in export.days],
                         [("camA", "2026-05-01"), ("camB", "2026-05-02")])
        day = export.day("camA", "2026-05-01")
        self.assertEqual({key: day[key] for key in ("events", "images", "people_unique", "adults", "children",
                                                    "age_unknown", "dogs", "dir_right", "dir_toward")},
                         {"events": "2", "images": "4", "people_unique": "3", "adults": "0", "children": "1",
                          "age_unknown": "2", "dogs": "1", "dir_right": "1", "dir_toward": "2"})
        # v2: the conservative MaxN total: walk event 1 + group event 2.
        self.assertEqual(day["people_max_frame"], "3")
        self.assertEqual(export.day_header.index("people_max_frame"), export.day_header.index("people_unique") - 1)
        other = export.day("camB", "2026-05-02")
        self.assertEqual((other["events"], other["images"], other["people_unique"], other["people_max_frame"]),
                         ("1", "1", "0", "0"))

    def test_run_json_reports_events_calibrations_and_settings(self):
        self.trail()
        export = self.run_export()
        summary = export.summary
        self.assertLessEqual(RUN_JSON_KEYS, set(summary))
        self.assertEqual(summary["version"], __version__)
        self.assertEqual(__version__, "2.0.0")
        self.assertEqual(summary["event_count"], 3)
        self.assertEqual(summary["images_with_postprocess_errors"], 0)
        self.assertEqual(summary["postprocess_settings"],
                         {"camera_id_mode": "folder", "camera_id_pattern": None, "event_gap_seconds": 60.0,
                          "near_fraction": 0.07, "events": True, "geometry_age": True, "large_bags": True})
        # v2: camB (no people) is reported too, as "insufficient" with n = 0.
        self.assertEqual(set(summary["camera_calibrations"]), {"camA|TestCam TC-1@640x480", "camB|TestCam TC-1@640x480"})
        calibration = summary["camera_calibrations"]["camA|TestCam TC-1@640x480"]
        self.assertEqual((calibration["status"], calibration["n"]), ("insufficient", 5))
        # Added: a per-camera near-zone diagnostic (walker x3, adult and child: all near).
        self.assertEqual(calibration["near_zone"], {"near_fraction": 0.07, "people_total": 5, "people_near": 5,
                                                    "status": "ok"})
        calibration = summary["camera_calibrations"]["camB|TestCam TC-1@640x480"]
        self.assertEqual((calibration["status"], calibration["n"]), ("insufficient", 0))
        self.assertEqual(calibration["near_zone"], {"near_fraction": 0.07, "people_total": 0, "people_near": 0,
                                                    "status": "ok"})
        # Added: the optional age expert is reported (off here).
        self.assertEqual((summary["age_model"], summary["age_mode"], summary["age_model_failures"]), (None, None, None))
        self.assertEqual(summary["export_subfolder"], export.folder.name)
        self.assertEqual(export.folder.name, "export_" + summary["run_id"])
        self.assertEqual(summary["input_dir"], str(self.images))
        self.assertEqual(summary["configuration"]["profile"], "standard")
        self.assertTrue(summary["configuration"]["empty_frame_gate"])
        self.assertEqual((summary["contact_sheet"], summary["contact_sheet_error"], summary["contact_sheet_selection"]),
                         (None, None, []))
        self.assertEqual({path.name for path in export.folder.iterdir()},
                         {summary["csv"], summary["events_csv"], summary["camera_days_csv"],
                          export.folder.name + "_run.json"})
        self.assertTrue(all(path.is_dir() for path in self.output.iterdir()))


class IncrementalRunTests(PipelineCase):
    def test_second_run_uses_cache_and_reproduces_postprocessed_rows(self):
        self.trail()
        first = self.run_export()
        cache_files = {path: path.read_bytes() for path in self.cache.rglob("*.json")}
        second = self.run_export()
        self.assertEqual((second.summary["analyzed_images"], second.summary["cached_images"]), (0, 5))
        self.assertEqual(len(self.vision.calls), 5)
        self.assertEqual(len(self.attributes.calls), 5)
        self.assertEqual(self.vision_constructor.call_count, 1)  # nothing to analyze: models stay unloaded
        self.assertEqual(second.summary["model_load_seconds"], 0)
        self.assertEqual(self.ensure.call_count, 2)
        self.assertNotEqual(first.folder, second.folder)
        self.assertNotEqual(first.summary["csv"], second.summary["csv"])
        self.assertTrue(all(row["cache_hit"] == "True" for row in second.rows))
        self.assertEqual([without(row) for row in first.rows], [without(row) for row in second.rows])
        self.assertEqual([without(row) for row in first.events], [without(row) for row in second.events])
        self.assertEqual([without(row) for row in first.days], [without(row) for row in second.days])
        self.assertEqual(first.summary["camera_calibrations"], second.summary["camera_calibrations"])
        self.assertEqual({path: path.read_bytes() for path in self.cache.rglob("*.json")}, cache_files)

    def test_changed_image_content_is_reanalyzed_and_events_recomputed(self):
        self.trail()
        first = self.run_export()
        old = first.row(WALK[2])
        old_cache = self.cache_path(first, WALK[2])
        # Same capture time, different clothing colour: new content, new SHA256.
        self.add(WALK[2], fakes.scene(fakes.walker(340, color=(230, 210, 30))), "2026:05:01 10:00:20")
        second = self.run_export()
        self.assertEqual((second.summary["analyzed_images"], second.summary["cached_images"]), (1, 4))
        self.assertEqual(self.vision.calls.count(WALK[2]), 2)
        self.assertEqual(len(self.vision.calls), 6)
        new = second.row(WALK[2])
        self.assertNotEqual(new["sha256"], old["sha256"])
        self.assertEqual(new["image_id"], "img_" + new["sha256"][:16])
        self.assertEqual([row["cache_hit"] for row in second.rows],
                         ["False" if row["relative_path"] == WALK[2] else "True" for row in second.rows])
        # Events are rebuilt from cached and new images together: the recoloured
        # walker in the last frame is not matched to the first two frames.
        event = second.event_of(WALK[0])
        self.assertEqual((event["image_count"], event["people_unique"], event["people_max_frame"]), ("3", "2", "1"))
        self.assertEqual(event["needs_review"], "True")
        self.assertEqual(json.loads(event["images"]), list(WALK))  # v2: relative paths, stable across content edits
        self.assertTrue(old_cache.is_file())  # old content stays cached (e.g. for a revert)
        self.assertTrue(self.cache_path(second, WALK[2]).is_file())

    def test_force_reanalyzes_and_export_new_only_limits_image_rows(self):
        self.add("one.jpg", fakes.scene(fakes.walker(60)), "2026:05:01 10:00:00")
        self.add("two.jpg", fakes.scene(), "2026:05:01 11:00:00")
        self.run_export()
        new_only = self.run_export("--export-new-only")
        self.assertEqual(new_only.rows, [])
        self.assertEqual(new_only.header, FIELDS)
        self.assertEqual((new_only.summary["csv_rows"], new_only.summary["cached_images"]), (0, 2))
        # Events are still built from the whole inventory.
        self.assertEqual((new_only.summary["event_count"], len(new_only.events)), (2, 2))
        forced = self.run_export("--force", "--export-new-only")
        self.assertEqual((forced.summary["analyzed_images"], forced.summary["cached_images"]), (2, 0))
        self.assertEqual([row["relative_path"] for row in forced.rows], ["one.jpg", "two.jpg"])
        self.assertEqual(len(self.vision.calls), 4)
        self.assertEqual(self.run_export("--export-new-only").rows, [])
        self.add("three.jpg", fakes.scene(fakes.walker(300)), "2026:05:01 12:00:00")
        latest = self.run_export("--export-new-only")
        self.assertEqual([row["relative_path"] for row in latest.rows], ["three.jpg"])
        self.assertEqual(latest.summary["event_count"], 3)

    def test_inference_fingerprint_covers_profile_and_gate_but_not_postprocessing(self):
        self.add("one.jpg", fakes.scene(fakes.walker(60)), "2026:05:01 10:00:00")
        standard = self.run_export()
        fast = self.run_export("--profile", "fast")
        ungated = self.run_export("--no-empty-frame-gate")
        again = self.run_export()
        regrouped = self.run_export("--event-gap", "5", "--camera-id", "single")
        self.assertEqual([export.summary["analyzed_images"] for export in (standard, fast, ungated, again, regrouped)],
                         [1, 1, 1, 0, 0])
        hashes = [export.summary["configuration_hash"] for export in (standard, fast, ungated, again, regrouped)]
        self.assertEqual(len(set(hashes[:3])), 3)
        self.assertEqual(hashes[3:], [hashes[0], hashes[0]])
        self.assertEqual(fast.row("one.jpg")["profile"], "fast")
        self.assertEqual(regrouped.row("one.jpg")["camera_id"], "all")

    def test_empty_input_writes_empty_exports_without_loading_models(self):
        export = self.run_export()
        self.ensure.assert_not_called()
        self.vision_constructor.assert_not_called()
        self.assertEqual((export.rows, export.events, export.days), ([], [], []))
        self.assertEqual((export.header, export.event_header, export.day_header), (FIELDS, EVENT_FIELDS, SUMMARY_FIELDS))
        self.assertEqual((export.summary["image_count"], export.summary["event_count"]), (0, 0))
        self.assertEqual(export.summary["camera_calibrations"], {})


class FailureHandlingTests(PipelineCase):
    def test_corrupt_image_is_exported_as_error_and_retried(self):
        self.add("ok.jpg", fakes.scene(fakes.walker(60)), "2026:05:01 10:00:00")
        broken = self.images / "broken.jpg"
        broken.write_bytes(b"not an image")
        export = self.run_export(expect=1)
        row = export.row("broken.jpg")
        self.assertEqual(row["status"], "error")
        self.assertIn("UnidentifiedImageError", row["error"])
        self.assertEqual((row["sha256"], row["image_id"]), (file_hash(broken), "img_" + file_hash(broken)[:16]))
        # Blank, not zero: the photo was never assessed.
        for field in ("people_total", "adults", "children", "dogs", "dir_left", "persons_json", "event_id", "camera_id"):
            self.assertEqual(row[field], "", field)
        self.assertEqual(export.row("ok.jpg")["status"], "ok")
        self.assertEqual((export.summary["images_with_errors"], export.summary["event_count"]), (1, 1))
        self.assertIn("ERROR broken.jpg", self.stderr.getvalue())
        again = self.run_export(expect=1)
        self.assertEqual((self.vision.calls.count("broken.jpg"), self.vision.calls.count("ok.jpg")), (2, 1))
        self.assertEqual((again.summary["analyzed_images"], again.summary["cached_images"]), (1, 1))
        self.add("broken.jpg", fakes.scene(), "2026:05:01 11:00:00")
        repaired = self.run_export()
        self.assertEqual(repaired.summary["images_with_errors"], 0)
        self.assertEqual(repaired.row("broken.jpg")["people_total"], "0")

    def test_partial_attribute_errors_keep_counts_and_are_retried(self):
        self.attributes.failed = {0}
        self.add("pair.jpg", fakes.scene(fakes.walker(100), fakes.walker(300, color=CHILD_COLOR)),
                 "2026:05:01 10:00:00")
        export = self.run_export(expect=1)
        row = export.row("pair.jpg")
        self.assertEqual(row["status"], "partial_error")
        self.assertIn("failed crop", row["error"])
        self.assertEqual(row["people_total"], "2")
        self.assertEqual(len(json.loads(row["persons_json"])), 2)
        self.assertTrue(row["event_id"])
        self.assertEqual(export.summary["images_with_errors"], 1)
        self.assertFalse(self.cache_path(export, "pair.jpg").exists())  # partial results are never cached
        self.run_export(expect=1)
        self.assertEqual(len(self.vision.calls), 2)
        self.attributes.failed = set()
        self.assertEqual(self.run_export().row("pair.jpg")["status"], "ok")
        self.assertEqual(self.run_export().summary["cached_images"], 1)
        self.assertEqual(len(self.vision.calls), 3)

    def test_corrupt_cache_structure_is_recomputed(self):
        self.add("one.jpg", fakes.scene(fakes.walker(60)), "2026:05:01 10:00:00")
        first = self.run_export()
        path = self.cache_path(first, "one.jpg")
        valid = json.loads(path.read_text(encoding="utf-8"))

        def variant(change):
            value = copy.deepcopy(valid)
            change(value)
            return json.dumps(value)

        cases = {
            "invalid_json": "{not json",
            "root_array": json.dumps(["unexpected root array"]),
            "missing_sections": json.dumps({"status": "ok", "sha256": valid["sha256"], "_cache_format": CACHE_FORMAT}),
            "old_cache_format": variant(lambda v: v.update(_cache_format=1)),
            "other_sha256": variant(lambda v: v.update(sha256="0" * 64)),
            "not_ok": variant(lambda v: v.update(status="partial_error")),
            "empty_person": variant(lambda v: v.update(persons=[{}])),
            "short_person_box": variant(lambda v: v["persons"][0].update(bbox_xyxy=[1, 2, 3])),
            "person_attribute_error": variant(lambda v: v["persons"][0]["attributes"].update(status="error")),
            "detections_not_list": variant(lambda v: v["experts"]["pose"].update(detections=None)),
            "missing_expert": variant(lambda v: v["experts"].pop("megadetector")),
            "metadata_not_dict": variant(lambda v: v.update(metadata=None)),
            "strip_not_dict": variant(lambda v: v.update(data_strip=[0, 0])),
            "float_width": variant(lambda v: v.update(width=640.0)),
            # v2: person boxes, member records and every expert detection are validated too.
            "person_xyxy_inverted": variant(lambda v: v["persons"][0].update(xyxy=[200.0, 300.0, 100.0, 100.0])),
            "person_xyxy_nan": variant(lambda v: v["persons"][0]["xyxy"].__setitem__(0, float("nan"))),
            "person_xyxy_missing": variant(lambda v: v["persons"][0].pop("xyxy")),
            "member_confidence_text": variant(
                lambda v: next(iter(v["persons"][0]["members"].values())).update(confidence="0.9")),
            "member_detection_index_float": variant(
                lambda v: next(iter(v["persons"][0]["members"].values())).update(detection_index=0.0)),
            "detection_label_missing": variant(lambda v: v["experts"]["yoloe"]["detections"][0].pop("label")),
            "detection_confidence_inf": variant(
                lambda v: v["experts"]["yolo26n"]["detections"][0].update(confidence=float("inf"))),
            "detection_confidence_bool": variant(
                lambda v: v["experts"]["megadetector"]["detections"][0].update(confidence=True)),
            "detection_xyxy_short": variant(lambda v: v["experts"]["pose"]["detections"][0].update(xyxy=[1, 2, 3])),
            "detection_not_dict": variant(lambda v: v["experts"]["yolo26n"]["detections"].append("person")),
        }
        self.assertTrue(all(valid["experts"][name]["detections"] for name in ("yoloe", "yolo26n", "megadetector", "pose")))
        for name, text in cases.items():
            with self.subTest(name):
                path.write_text(text, encoding="utf-8")
                calls = len(self.vision.calls)
                export = self.run_export()
                self.assertEqual((export.summary["analyzed_images"], len(self.vision.calls)), (1, calls + 1))
                row = export.row("one.jpg")
                self.assertEqual((row["status"], row["people_total"], row["cache_hit"]), ("ok", "1", "False"))
                repaired = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual((repaired["_cache_format"], len(repaired["persons"])), (CACHE_FORMAT, 1))

    def test_postprocessing_failure_of_one_image_does_not_abort_export(self):
        self.add("good.jpg", fakes.scene(fakes.walker(60)), "2026:05:01 10:00:00")
        self.add("bad.jpg", fakes.scene(fakes.walker(300), corrupt_postprocess=True), "2026:05:01 10:00:05")
        export = self.run_export("--contact-sheet", expect=1)
        bad = export.row("bad.jpg")
        self.assertEqual(bad["status"], "error")
        # v2: the message names the failing stage.
        self.assertEqual(bad["error"], "Post-processing failed: fusion: KeyError: 'detection_index'")
        self.assertEqual((bad["people_total"], bad["event_id"], bad["persons_json"]), ("", "", ""))
        good = export.row("good.jpg")
        self.assertEqual((good["status"], good["people_total"]), ("ok", "1"))
        summary = export.summary
        self.assertEqual((summary["images_with_postprocess_errors"], summary["images_with_errors"]), (1, 1))
        self.assertEqual(summary["event_count"], 1)
        self.assertEqual(json.loads(export.events[0]["images"]), ["good.jpg"])  # v2: relative paths
        self.assertIsNone(summary["contact_sheet_error"])
        self.assertEqual(len(summary["contact_sheet_selection"]), 2)
        # Inference itself succeeded and stays cached; only post-processing is repeated.
        again = self.run_export(expect=1)
        self.assertEqual((again.summary["cached_images"], again.summary["images_with_postprocess_errors"]), (2, 1))

    def test_active_lock_blocks_run_before_model_setup(self):
        self.add("one.jpg")
        self.cache.mkdir()
        lock = self.cache / "run.lock"
        lock.write_text("another process")
        with self.assertRaises(RuntimeError):
            entry.run(self.args())
        self.ensure.assert_not_called()
        self.assertEqual(lock.read_text(), "another process")
        self.assertEqual(entry.main(["--input", str(self.images), "--output", str(self.output), "--models",
                                     str(self.models), "--cache", str(self.cache)]), 2)

    def test_lock_is_released_after_success_and_failure(self):
        self.add("one.jpg", fakes.scene(fakes.walker(60)))
        self.run_export()
        self.assertFalse((self.cache / "run.lock").exists())
        # __main__ imports postprocess at module level (v2), so patch its reference there.
        with patch("trailcam.__main__.postprocess", side_effect=RuntimeError("boom")) as failing:
            with self.assertRaisesRegex(RuntimeError, "boom"):
                entry.run(self.args())
        failing.assert_called_once()
        self.assertFalse((self.cache / "run.lock").exists())


class OptionTests(PipelineCase):
    def test_defaults_and_settings_file(self):
        args = entry.arguments([])
        self.assertEqual((args.contact_sheet, args.contact_sheet_size, args.event_gap, args.camera_id_mode,
                          args.camera_pattern, args.profile, args.empty_frame_gate, args.device, args.threads),
                         (True, 20, 60, "folder", None, "standard", False, "auto", 8))
        self.assertEqual(args.input, self.images)
        (self.root / "settings.json").write_text(json.dumps({
            "event_gap_seconds": 15, "camera_id_mode": "regex", "camera_id_pattern": r"(?P<camera>cam\d+)",
            "profile": "fast", "contact_sheet": False}), encoding="utf-8")
        args = entry.arguments([])
        self.assertEqual((args.event_gap, args.camera_id_mode, args.camera_pattern, args.profile, args.contact_sheet),
                         (15, "regex", r"(?P<camera>cam\d+)", "fast", False))
        args = entry.arguments(["--event-gap", "30", "--camera-id", "folder", "--contact-sheet"])
        self.assertEqual((args.event_gap, args.camera_id_mode, args.contact_sheet), (30.0, "folder", True))

    def test_engines_receive_cli_options_and_the_decoded_image(self):
        self.add("half.jpg", fakes.scene(fakes.walker(100), decode_scale=2), "2026:05:01 10:00:00")
        export = self.run_export("--device", "cpu", "--threads", "2", "--profile", "fast", "--no-empty-frame-gate")
        self.vision_constructor.assert_called_once_with(self.models, "cpu", 2, "fast", False)
        self.attribute_constructor.assert_called_once_with(self.models, "cpu", 2)
        call, = self.attributes.calls
        # The already decoded (draft, half size) image is reused with its scale.
        self.assertEqual((call["name"], call["image_size"], call["scale"], call["person_ids"]),
                         ("half.jpg", (320, 240), 2.0, ["cand_0001"]))
        cached = json.loads(self.cache_path(export, "half.jpg").read_text(encoding="utf-8"))
        self.assertNotIn("_decoded_image", cached)
        self.assertEqual(cached["analysis_scale"], 2.0)
        row = export.row("half.jpg")
        self.assertEqual((row["profile"], row["vision_device"], row["attribute_device"], row["gated_empty"]),
                         ("fast", "cpu", "cpu", "False"))
        configuration_used = export.summary["configuration"]
        self.assertEqual((configuration_used["profile"], configuration_used["empty_frame_gate"],
                          configuration_used["requested_device"], configuration_used["threads"]),
                         ("fast", False, "cpu", 2))

    def test_camera_id_regex_mode_groups_by_named_group(self):
        for x, relative in zip((60, 160, 260), ("site1/cam01_IMG_0001.jpg", "site1/cam02_IMG_0002.jpg",
                                                "site1/misc_IMG_0003.jpg")):
            self.add(relative, fakes.scene(fakes.walker(x)), "2026:05:01 10:00:00")
        folder = self.run_export()
        self.assertEqual({row["camera_id"] for row in folder.rows}, {"site1"})
        self.assertEqual(folder.summary["event_count"], 1)
        self.assertEqual(folder.events[0]["people_unique"], "1")
        regex = self.run_export("--camera-id", "regex", "--camera-pattern", r"(?P<camera>cam\d+)_")
        self.assertEqual([row["camera_id"] for row in regex.rows], ["cam01", "cam02", "site1"])  # no match -> folder
        self.assertEqual(regex.summary["cached_images"], 3)  # camera grouping is post-processing only
        self.assertEqual(regex.summary["postprocess_settings"]["camera_id_mode"], "regex")
        self.assertEqual(regex.summary["postprocess_settings"]["camera_id_pattern"], r"(?P<camera>cam\d+)_")
        self.assertEqual(regex.summary["event_count"], 3)
        for row in regex.rows:
            self.assertTrue(row["event_id"].startswith(row["camera_id"] + "__"), row["event_id"])
        self.assertEqual({key.split("|")[0] for key in regex.summary["camera_calibrations"]}, {"cam01", "cam02", "site1"})
        single = self.run_export("--camera-id", "single")
        self.assertEqual({row["camera_id"] for row in single.rows}, {"all"})
        self.assertEqual(single.summary["event_count"], 1)

    def test_invalid_camera_pattern_is_rejected_before_any_work(self):
        cases = ((["--camera-id", "regex", "--camera-pattern", "(?P<camera>["], "Invalid --camera-pattern"),
                 (["--camera-id", "regex", "--camera-pattern", r"cam\d+"], "named group"),
                 (["--camera-id", "regex"], "named group"),
                 (["--camera-id", "bogus"], "invalid choice"))
        for extra, message in cases:
            with self.subTest(extra=extra):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    self.args(*extra)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
        self.ensure.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_event_gap_must_be_a_positive_number(self):
        # v2: non-finite gaps (nan, inf) are refused as well; the message says so.
        finite = "--event-gap must be a finite positive number of seconds."
        for value, message in (("0", finite), ("-5", finite), ("nan", finite), ("inf", finite),
                               ("-inf", finite), ("abc", "invalid float value")):
            with self.subTest(value=value):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    self.args(f"--event-gap={value}")
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
        self.assertEqual(self.args("--event-gap", "0.5").event_gap, 0.5)
        self.ensure.assert_not_called()

    def test_settings_file_values_are_validated_before_any_work(self):
        # v2: settings.json values become argparse defaults, which argparse does
        # not check against ``choices`` or ``type``; they are validated explicitly.
        cases = (({"profile": "turbo"}, "Invalid profile 'turbo'"),
                 ({"camera_id_mode": "per_card"}, "Invalid camera_id_mode 'per_card'"),
                 ({"device": "gpu0"}, "Invalid device 'gpu0'"),
                 ({"threads": 2.5}, "threads must be an integer."),
                 ({"threads": True}, "threads must be an integer."),
                 ({"contact_sheet_size": "many"}, "invalid int value"),
                 ({"recursive": "yes"}, "recursive must be true or false."),
                 ({"empty_frame_gate": 1}, "empty_frame_gate must be true or false."),
                 ({"event_gap_seconds": float("nan")}, "--event-gap must be a finite positive number of seconds."),
                 ({"event_gap_seconds": float("inf")}, "--event-gap must be a finite positive number of seconds."),
                 ({"event_gap_seconds": 0}, "--event-gap must be a finite positive number of seconds."),
                 ({"event_gap_seconds": [60]}, "--event-gap must be a number of seconds."))
        self.add("one.jpg", fakes.scene(fakes.walker(60)), "2026:05:01 10:00:00")
        for values, message in cases:
            with self.subTest(values=values):
                (self.root / "settings.json").write_text(json.dumps(values), encoding="utf-8")  # NaN -> JSON NaN
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    entry.main(["--input", str(self.images), "--output", str(self.output), "--models",
                                str(self.models), "--cache", str(self.cache)])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
        self.ensure.assert_not_called()
        self.vision_constructor.assert_not_called()
        self.assertEqual(self.vision.calls, [])
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.output.exists())
        # A command-line option does not rescue an invalid settings.json value it does not replace...
        (self.root / "settings.json").write_text(json.dumps({"profile": "turbo"}), encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("--camera-id", "single")
        # ...but one it replaces does.
        self.assertEqual(self.args("--profile", "fast").profile, "fast")

    def test_event_gap_splits_events_without_reanalysis(self):
        self.add("IMG_0001.jpg", fakes.scene(fakes.walker(60)), "2026:05:01 10:00:00")
        self.add("IMG_0002.jpg", fakes.scene(fakes.walker(200)), "2026:05:01 10:00:30")
        joined = self.run_export()
        self.assertEqual((joined.summary["event_count"], joined.events[0]["people_unique"]), (1, "1"))
        self.assertEqual({row["event_image_count"] for row in joined.rows}, {"2"})
        split = self.run_export("--event-gap", "10")
        self.assertEqual(split.summary["postprocess_settings"]["event_gap_seconds"], 10.0)
        self.assertEqual((split.summary["event_count"], split.summary["cached_images"]), (2, 2))
        self.assertEqual({row["event_image_count"] for row in split.rows}, {"1"})
        self.assertEqual(sorted(event["people_unique"] for event in split.events), ["1", "1"])

    def test_contact_sheet_disabled_skips_renderer(self):
        self.add("one.jpg", fakes.scene(fakes.walker(60)))
        with patch("trailcam.contacts.make_contact_sheet") as render:
            export = self.run_export("--no-contact-sheet")
        render.assert_not_called()
        self.assertEqual((export.summary["contact_sheet"], export.summary["contact_sheet_selection"]), (None, []))
        self.assertFalse(list(export.folder.glob("*.jpg")))

    def test_contact_sheet_renders_selected_tiny_images(self):
        self.trail()
        export = self.run_export("--contact-sheet", "--contact-sheet-size", "3")
        summary = export.summary
        self.assertIsNone(summary["contact_sheet_error"])
        self.assertEqual(summary["contact_sheet"], export.folder.name + "_contactsheet.jpg")
        selection = summary["contact_sheet_selection"]
        self.assertEqual([panel["panel"] for panel in selection], [1, 2, 3])
        self.assertEqual(len({panel["image_id"] for panel in selection}), 3)
        self.assertLessEqual({panel["image_id"] for panel in selection}, {row["image_id"] for row in export.rows})
        self.assertEqual(len(list(export.folder.iterdir())), 5)
        with Image.open(export.folder / summary["contact_sheet"]) as image:
            self.assertEqual(image.format, "JPEG")

    def test_contact_sheet_failure_keeps_the_csv_exports(self):
        self.add("one.jpg", fakes.scene(fakes.walker(60)))
        with patch("trailcam.contacts.make_contact_sheet", side_effect=OSError("disk full")):
            export = self.run_export("--contact-sheet", expect=1)
        self.assertEqual(export.summary["contact_sheet_error"], "OSError: disk full")
        self.assertIsNone(export.summary["contact_sheet"])
        self.assertEqual(len(export.rows), 1)
        self.assertIn("Contact sheet failed", self.stderr.getvalue())

    def test_recursive_discovery_excludes_output_models_and_hidden(self):
        self.add("root.jpg")
        self.add("child/child.jpg")
        self.add(".hidden/secret.jpg")
        nested_output = self.images / "exports"
        nested_output.mkdir()
        Image.new("RGB", (20, 20)).save(nested_output / "old_contactsheet.jpg")
        discovered = entry.discover(self.images, excluded=(nested_output,))
        self.assertEqual({path.relative_to(self.images).as_posix() for path in discovered},
                         {"root.jpg", "child/child.jpg"})
        self.assertEqual([path.name for path in entry.discover(self.images, recursive=False)], ["root.jpg"])


class ReviewFixTests(PipelineCase):
    """Regression tests for the v2 code-review fixes, end to end."""

    def test_byte_identical_copies_in_two_folders_are_separate_appearances(self):
        original = self.add("camA/IMG_0001.jpg", fakes.scene(fakes.walker(60)), "2026:05:01 10:00:00")
        duplicate = self.images / "camB" / "IMG_0001.jpg"
        duplicate.parent.mkdir()
        shutil.copyfile(original, duplicate)
        export = self.run_export()
        a, b = export.row("camA/IMG_0001.jpg"), export.row("camB/IMG_0001.jpg")
        self.assertEqual((a["sha256"], a["image_id"]), (b["sha256"], b["image_id"]))  # one content, one image_id
        self.assertEqual(self.vision.calls, ["camA/IMG_0001.jpg"])  # the copy reuses the cached inference
        self.assertEqual((export.summary["analyzed_images"], export.summary["cached_images"]), (1, 1))
        self.assertEqual((a["camera_id"], b["camera_id"]), ("camA", "camB"))
        self.assertEqual(len(export.events), 2)
        self.assertNotEqual(a["event_id"], b["event_id"])
        tracks = set()
        for row in (a, b):
            with self.subTest(relative=row["relative_path"]):
                event = export.event_of(row["relative_path"])
                self.assertEqual(json.loads(event["images"]), [row["relative_path"]])
                self.assertEqual((event["image_count"], event["people_max_frame"], event["people_unique"]),
                                 ("1", "1", "1"))
                self.assertEqual((row["event_frame"], row["event_image_count"], row["people_total"]), ("1", "1", "1"))
                person, = json.loads(row["persons_json"])
                self.assertTrue(person["track"].startswith(row["event_id"] + "__t"), person["track"])
                self.assertEqual(person["direction_source"], "facing")
                tracks.add(person["track"])
        self.assertEqual(len(tracks), 2)
        self.assertEqual(sorted((d["camera_id"], d["people_max_frame"]) for d in export.days),
                         [("camA", "1"), ("camB", "1")])

    def test_byte_identical_copies_in_one_folder_are_two_frames_of_one_event(self):
        original = self.add("camA/IMG_0001.jpg", fakes.scene(fakes.walker(60)), "2026:05:01 10:00:00")
        shutil.copyfile(original, self.images / "camA" / "IMG_0001 (1).jpg")
        # The copy sorts first and is the one analysed; the fake keys scenes by path.
        self.scenes["camA/IMG_0001 (1).jpg"] = self.scenes["camA/IMG_0001.jpg"]
        export = self.run_export()
        self.assertEqual(self.vision.calls, ["camA/IMG_0001 (1).jpg"])
        event, = export.events
        self.assertEqual(json.loads(event["images"]), ["camA/IMG_0001 (1).jpg", "camA/IMG_0001.jpg"])
        self.assertEqual((event["image_count"], event["people_max_frame"], event["people_unique"]), ("2", "1", "1"))
        rows = [export.row(relative) for relative in json.loads(event["images"])]
        self.assertEqual([row["event_frame"] for row in rows], ["1", "2"])
        tracks = {json.loads(row["persons_json"])[0]["track"] for row in rows}
        self.assertEqual(len(tracks), 1)
        self.assertTrue(next(iter(tracks)).startswith(event["event_id"]))

    def test_event_ids_are_stable_when_an_unrelated_photo_is_added(self):
        self.trail()
        first = self.run_export()
        # A camera that sorts before all others, and an earlier separate event on camA.
        self.add("cam0/IMG_0001.jpg", fakes.scene(fakes.walker(60)), "2026:04:30 09:00:00")
        self.add("camA/IMG_0000.jpg", fakes.scene(), "2026:05:01 08:00:00")
        second = self.run_export()
        self.assertEqual((len(first.events), len(second.events)), (3, 5))
        for relative in TRAIL:
            with self.subTest(relative=relative):
                self.assertEqual(second.row(relative)["event_id"], first.row(relative)["event_id"])
                self.assertEqual([p["track"] for p in json.loads(second.row(relative)["persons_json"])],
                                 [p["track"] for p in json.loads(first.row(relative)["persons_json"])])
        self.assertEqual(first.row(WALK[0])["event_id"], "camA__2026-05-01T10-00-00")
        self.assertEqual(second.row("cam0/IMG_0001.jpg")["event_id"], "cam0__2026-04-30T09-00-00")
        old = {event["event_id"]: without(event) for event in first.events}
        new = {event["event_id"]: without(event) for event in second.events}
        self.assertEqual({key: new[key] for key in old}, old)

    def test_metadata_is_refreshed_on_a_cache_hit_for_a_renamed_copy(self):
        original = self.add("camA/IMG_0001.jpg", fakes.scene(fakes.walker(60)))  # no EXIF time: file date
        first_time = datetime(2026, 5, 1, 10, 0, 0)
        os.utime(original, (first_time.timestamp(),) * 2)
        first = self.run_export()
        self.assertEqual({key: first.row("camA/IMG_0001.jpg")[key] for key in ("capture_time", "time_source", "sequence_number")},
                         {"capture_time": "2026-05-01T10:00:00", "time_source": "file_mtime", "sequence_number": "1"})
        renamed = self.images / "camC" / "IMAG0042.jpg"
        renamed.parent.mkdir()
        shutil.copyfile(original, renamed)
        os.utime(renamed, (datetime(2026, 5, 3, 15, 30, 0).timestamp(),) * 2)
        second = self.run_export()
        self.assertEqual((second.summary["analyzed_images"], second.summary["cached_images"]), (0, 2))
        self.assertEqual(self.vision.calls, ["camA/IMG_0001.jpg"])
        row = second.row("camC/IMAG0042.jpg")
        self.assertEqual(row["cache_hit"], "True")
        # v2: metadata is re-read from the file itself, not taken from the cached original.
        self.assertEqual((row["capture_time"], row["time_source"], row["sequence_number"]),
                         ("2026-05-03T15:30:00", "file_mtime", "42"))
        self.assertEqual(second.event_of("camC/IMAG0042.jpg")["start"], "2026-05-03T15:30:00")
        self.assertEqual(second.row("camA/IMG_0001.jpg")["capture_time"], "2026-05-01T10:00:00")
        cached = json.loads(self.cache_path(second, "camC/IMAG0042.jpg").read_text(encoding="utf-8"))
        self.assertEqual(cached["metadata"]["sequence_number"], 1)  # the cache itself is not rewritten

    def test_gated_frame_has_no_candidates_and_blank_skipped_expert_counts(self):
        # A faint person seen only by YOLO26n (above the raw floor, below the gate).
        self.add("faint.jpg", fakes.scene(fakes.person([200, 100, 280, 300], {"yolo26n": .15})), "2026:05:01 10:00:00")
        export = self.run_export()
        row = export.row("faint.jpg")
        self.assertEqual((row["gated_empty"], row["people_total"], row["candidates_total"], row["candidates_rejected"],
                          row["persons_json"]), ("True", "0", "0", "0", "[]"))
        for field in ("people_total", "bicycles", "strollers", "dogs", "atv_utv"):
            self.assertEqual(row["yoloe_" + field], "", field)  # skipped: not assessed
        self.assertEqual((row["yolo26n_people_total"], row["megadetector_people_total"]), ("0", "0"))
        self.assertEqual(row["vote_status_strollers"], "no_eligible_output")
        cached = json.loads(self.cache_path(export, "faint.jpg").read_text(encoding="utf-8"))
        self.assertEqual(cached["persons"], [])
        self.assertEqual([d["confidence"] for d in cached["experts"]["yolo26n"]["detections"]], [.15])
        self.assertEqual(cached["experts"]["pose"]["skipped"], "empty_frame_gate")
        # With the gate off the faint box is a (rejected) candidate again.
        ungated = self.run_export("--no-empty-frame-gate").row("faint.jpg")
        self.assertEqual((ungated["gated_empty"], ungated["candidates_total"], ungated["candidates_rejected"],
                          ungated["people_total"], ungated["yoloe_people_total"]), ("False", "1", "1", "0", "0"))

    def test_row_that_cannot_be_written_becomes_an_error_row(self):
        self.add("good.jpg", fakes.scene(fakes.walker(60)), "2026:05:01 10:00:00")
        self.add("odd.jpg", fakes.scene(fakes.walker(300)), "2026:05:01 10:00:05")
        real = entry.make_row

        def fragile(result, metadata):
            if metadata["relative_path"] == "odd.jpg" and result.get("status") == "ok":
                raise ValueError("cannot serialise")
            return real(result, metadata)

        with patch.object(entry, "make_row", side_effect=fragile):
            export = self.run_export(expect=1)
        odd = export.row("odd.jpg")
        self.assertEqual((odd["status"], odd["error"]), ("error", "Export failed: ValueError: cannot serialise"))
        self.assertEqual((odd["people_total"], odd["persons_json"], odd["event_id"]), ("", "", ""))
        self.assertEqual(odd["sha256"], file_hash(self.images / "odd.jpg"))
        self.assertEqual(export.row("good.jpg")["status"], "ok")
        self.assertEqual((export.summary["images_with_errors"], export.summary["csv_rows"]), (1, 2))
        event, = export.events  # post-processing ran before the row failed: the photo is in its event
        self.assertEqual((json.loads(event["images"]), event["image_count"]), (["good.jpg", "odd.jpg"], "2"))
        self.assertTrue(self.cache_path(export, "odd.jpg").is_file())

    def test_console_streams_escape_unencodable_file_names(self):
        # A legacy Windows code page cannot encode Hebrew; printing progress must not abort the run.
        self.add("עין גדי/IMG_0001.jpg", fakes.scene(fakes.walker(60)), "2026:05:01 10:00:00")
        raw_out, raw_err = io.BytesIO(), io.BytesIO()
        out = io.TextIOWrapper(raw_out, encoding="cp1252", errors="strict", write_through=True)
        err = io.TextIOWrapper(raw_err, encoding="cp1252", errors="strict", write_through=True)
        with patch.object(sys, "stdout", out), patch.object(sys, "stderr", err):
            code = entry.main(["--input", str(self.images), "--output", str(self.output), "--models",
                               str(self.models), "--cache", str(self.cache), "--no-contact-sheet"])
        self.assertEqual(code, 0, raw_err.getvalue())
        self.assertEqual((out.errors, err.errors), ("backslashreplace", "backslashreplace"))
        self.assertIn(b"ok: \\u05e2\\u05d9\\u05df", raw_out.getvalue())
        export = Export(next(self.output.glob("export_*")))
        self.assertEqual(export.row("עין גדי/IMG_0001.jpg")["people_total"], "1")


class LargeImageTests(PipelineCase):
    def test_sixty_person_image_preserves_all_native_evidence_in_csv(self):
        people = []
        for index in range(60):
            x, y = 40 + 195 * (index % 10), 40 + 210 * (index // 10)
            color = (40 + index * 37 % 200, 40 + index * 91 % 200, 40 + index * 53 % 200)
            people.append(fakes.person([x, y, x + 100, y + 180], color=color))
            if index < 20:  # faint, rejected candidates also stay out of persons_json
                people.append(fakes.person([x + 105, y + 20, x + 145, y + 100], {"yolo26n": .15}, color=(128, 128, 128)))
        self.add("crowd.jpg", fakes.scene(*people, size=(2000, 1320)), "2026:05:01 10:00:00")
        export = self.run_export()
        row = export.row("crowd.jpg")
        self.assertEqual((row["people_total"], row["candidates_total"], row["candidates_rejected"]), ("60", "80", "20"))
        self.assertEqual(len(json.loads(row["persons_json"])), 60)
        self.assertEqual(export.events[0]["people_unique"], "60")
        for name, rows in (("images", export.rows), ("events", export.events), ("camera_days", export.days)):
            for record in rows:
                for field, value in record.items():
                    if field != "persons_json":
                        self.assertLess(len(value), EXCEL_CELL_LIMIT, f"{name}.{field}")
        # Crowded scenes can exceed Excel's cell limit; CSV readers retain all evidence.
        native = json.loads(row["persons_json"])
        self.assertTrue(all(p["attributes"]["scores"] for p in native))


# ----------------------------------------------------------------- unit-level checks

class _Tensor:
    """Stands in for a torch tensor: ``.cpu()`` and ``.tolist()``."""

    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.values)


class _StubModel:
    """Ultralytics-like model returning the scene's raw detections."""

    NAMES = {"yoloe": fakes.YOLOE_NAMES, "yolo26n": fakes.YOLO26N_NAMES,
             "megadetector": dict(fakes.MD_LABELS), "pose": fakes.POSE_NAMES}

    def __init__(self, spec, key):
        self.spec, self.key = spec, key

    def predict(self, image, device=None, **options):
        height, width = image.shape[:2]
        detections = fakes.expert_detections(self.spec, self.key, width, height, PROFILES["standard"]["pose"])
        boxes = _Tensor(None)
        boxes.xyxy = _Tensor([d["xyxy"] for d in detections])
        boxes.cls = _Tensor([float(d["class_id"]) for d in detections])
        boxes.conf = _Tensor([d["confidence"] for d in detections])
        result = SimpleNamespace(boxes=boxes, names=self.NAMES[self.key], speed=dict(fakes.SPEED_MS),
                                 masks=None, keypoints=None)
        if self.key == "yoloe" and detections:
            result.masks = SimpleNamespace(xy=[np.array(d["mask_polygon_xy"], dtype=float) for d in detections])
        if self.key == "pose":
            result.keypoints = SimpleNamespace(data=_Tensor([d["keypoints"] for d in detections]))
        return [result]


def comparable(value):
    if isinstance(value, dict):
        return {key: comparable(item) for key, item in value.items() if key != "inference_seconds"}
    if isinstance(value, (list, tuple)):
        return [comparable(item) for item in value]
    if isinstance(value, float):
        return round(value, 9)
    return value


class FakeContractTests(unittest.TestCase):
    def test_fake_results_match_the_real_vision_engine_output(self):
        """The fakes must track VisionEngine.analyze, or the pipeline tests test nothing real."""
        specs = {
            "people.jpg": fakes.scene(
                fakes.walker(60), fakes.walker(260, y=170, height=130, facing="away", color=CHILD_COLOR),
                fakes.person([520, 60, 560, 160], {"yolo26n": .15}),
                fakes.thing("dog", [450, 250, 560, 320], {"yoloe": .8, "yolo26n": .7, "megadetector": .6}),
                fakes.thing("backpack", [75, 140, 125, 200], {"yoloe": .6, "yolo26n": .3}),
                fakes.thing("bicycle", [350, 300, 440, 380], {"yoloe": .7, "yolo26n": .6, "megadetector": .4})),
            "gated.jpg": fakes.scene(fakes.person([300, 460, 380, 478], {"yolo26n": .9}), strip_rows=24),
            # Above the raw floor, below the gate: v2 builds no candidate for a gated frame.
            "faint.jpg": fakes.scene(fakes.person([200, 100, 280, 300], {"yolo26n": .15})),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, spec in specs.items():
                path = fakes.write_image(Path(tmp) / name, spec, "2026:05:01 10:00:00")
                engine = object.__new__(VisionEngine)
                engine.torch = SimpleNamespace(get_num_threads=lambda: 8, set_num_threads=lambda n: None)
                engine.threads, engine.device, engine.warnings = 8, "cpu", []
                engine.profile, engine.sizes, engine.empty_frame_gate = "standard", dict(PROFILES["standard"]), True
                engine.models = {key: _StubModel(spec, key) for key in fakes.EXPERTS}
                real = engine.analyze(path)
                fake = fakes.FakeVisionEngine({name: spec}).analyze(path)
                with self.subTest(name):
                    self.assertEqual(real.pop("_decoded_image").size, fake.pop("_decoded_image").size)
                    self.assertEqual(set(real.pop("timings")), set(fake.pop("timings")))
                    self.assertEqual(comparable(real), comparable(fake))
                    self.assertEqual(real["gated_empty"], name in ("gated.jpg", "faint.jpg"))
                    if name == "faint.jpg":
                        self.assertEqual(len(real["experts"]["yolo26n"]["detections"]), 1)
                        self.assertEqual(real["persons"], [])
                    if name == "people.jpg":
                        self.assertEqual(len(real["persons"]), 3)
                        facing = {p["orientation_evidence"]["facing_direction"] for p in real["persons"]
                                  if p["orientation_evidence"]}
                        self.assertEqual(facing, {"toward", "away"})


class PostprocessUnitTests(unittest.TestCase):
    def test_contact_selection_is_diverse_unique_and_reproducible(self):
        items = []
        extras = {29: fakes.thing("dog", [450, 250, 560, 320], {"yoloe": .8, "yolo26n": .7}),
                  32: fakes.thing("baby stroller", [450, 250, 540, 330], {"yoloe": .8}),
                  34: fakes.thing("all-terrain vehicle", [400, 250, 600, 380], {"yoloe": .8})}
        for index in range(35):
            spec = fakes.scene(fakes.walker(60 + index % 5 * 40), *([extras[index]] if index in extras else []))
            when = f"2026-05-01T{10 + index // 6:02d}:{index % 6 * 10:02d}:00"
            result = fakes.with_attributes(fakes.result_for(spec, {"capture_time": when}))
            items.append({"relative_path": f"frame_{index:03d}.jpg", "cache_hit": index % 2 == 0,
                          "result": result, "image_id": f"img{index:03d}", "sha256": ""})
        postprocess(items, {"events": True, "geometry_age": True, "large_bags": True, "near_fraction": .07})
        first = select_examples(items, 20)
        second = select_examples(list(reversed(items)), 20)
        self.assertEqual(len(first), 20)
        self.assertEqual(len({item["image_id"] for item in first}), 20)
        self.assertEqual([item["image_id"] for item in first], [item["image_id"] for item in second])
        self.assertTrue(all(not item["cache_hit"] for item in first[:17]))  # new images first
        covered = set().union(*(features(item) for item in first))
        self.assertLessEqual({"dogs", "strollers", "atv_utv", "one_person"}, covered)

    def test_orientation_and_backpack_conflicts_abstain(self):
        spec = fakes.scene(fakes.walker(100), fakes.thing("backpack", [115, 140, 165, 200], {"yoloe": .8}))
        result = fakes.with_attributes(fakes.result_for(spec))
        person = result["persons"][0]
        self.assertEqual(person["orientation_evidence"]["orientation"], "front")
        person["attributes"]["native_orientation_label"] = "back"
        person["attributes"]["backpack_presence"] = False
        fuse(result)
        combined = person["combined"]
        self.assertEqual((combined["orientation"], combined["orientation_source"], combined["facing"]),
                         ("unknown", "conflict_abstention", "unclear"))
        self.assertIsNone(combined["carrying_backpack"])
        self.assertEqual(combined["backpack_source"], "attribute_detector_conflict")
        self.assertEqual(result["combined"]["orientation_conflicts"], 1)
        self.assertTrue(result["needs_review"])

    def test_inference_configuration_tracks_models_profile_gate_and_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            model = models / "test.pt"
            model.write_bytes(b"first weights")
            first, settings = configuration(models, "auto", 8)
            self.assertEqual(settings["schema_version"], CACHE_FORMAT)
            self.assertEqual(settings["candidate_policy"]["expert_order"], list(roster.EXPERT_ORDER))
            self.assertEqual(configuration(models, "auto", 2)[0], first)  # threads never change outputs
            variants = [configuration(models, "auto", 8, "fast")[0], configuration(models, "auto", 8, "standard", True)[0],
                        configuration(models, "cpu", 8)[0]]
            # v2: the roster's expert order picks each candidate's representative box (cached).
            with patch.object(roster, "EXPERT_ORDER", ("pose", "yoloe", "yolo26n")):
                variants.append(configuration(models, "auto", 8)[0])
            self.assertEqual(configuration(models, "auto", 8)[0], first)
            model.write_bytes(b"changed weights")
            variants.append(configuration(models, "auto", 8)[0])
        self.assertEqual(len({first, *variants}), 6)

    def test_csv_filename_formula_is_escaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "export.csv"
            write_csv(path, [{"relative_path": "=malicious.jpg", "count": 0}], ["relative_path", "count"])
            rows, _ = read_csv(path)
        self.assertEqual(rows[0]["relative_path"], "'=malicious.jpg")
        self.assertEqual(rows[0]["count"], "0")

    def test_attribute_cuda_failure_records_cpu_after_fallback(self):
        import cv2
        engine = object.__new__(AttributeEngine)
        engine.device = "cuda:0"
        engine.requested_device = "auto"
        engine.fallback_reasons = []
        engine._np = np
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
            result = engine.analyze(image_path, [{"person_id": "cand_0001", "bbox_xyxy": [0, 0, 100, 200]}])
        self.assertEqual(calls, ["cuda:0", "cpu"])
        self.assertEqual(engine.device, "cpu")
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["persons"][0]["attributes"]["device"], "cpu")
        self.assertEqual(result["persons"][0]["attributes"]["scores"]["native_female"], 0.0)
        self.assertIn("CUDA inference failed", engine.fallback_reasons[0])


class NearZoneOptionTests(PipelineCase):
    def args(self, *extra):
        args = super().args(*extra)
        if not any(value.startswith("--near-fraction") for value in extra) and (self.root / "settings.json").exists():
            args.near_fraction = entry.arguments([]).near_fraction
        return args

    """``--near-fraction``: the near zone is post-processing only (no re-analysis)."""

    def add_near_and_far(self):
        # A 200 px walker (near at 0.07 * 480 = 33.6 px) and a 20 px person far up the frame.
        self.add("camA/IMG_0001.jpg", fakes.scene(fakes.walker(60), fakes.person([500, 40, 508, 60], color=(90, 90, 200))),
                 "2026:05:01 10:00:00")

    def test_near_columns_in_all_three_csvs(self):
        self.add_near_and_far()
        export = self.run_export()
        row = export.row("camA/IMG_0001.jpg")
        self.assertEqual((row["people_total"], row["people_near"], row["people_class"], row["people_near_class"]),
                         ("2", "1", "2", "1"))
        people = sorted(json.loads(row["persons_json"]), key=lambda p: p["box"][0])
        self.assertEqual([(p["near"], p["near_source"]) for p in people],
                         [(True, "height_at_similar_depth"), (False, "height_at_similar_depth")])
        event = export.event_of("camA/IMG_0001.jpg")
        self.assertEqual((event["people_unique"], event["people_near_max_frame"], event["people_near_unique"],
                          event["people_unique_class"], event["people_near_unique_class"]), ("2", "1", "1", "2", "1"))
        day = export.day("camA", "2026-05-01")
        self.assertEqual((day["people_near_max_frame"], day["people_near_unique"]), ("1", "1"))
        self.assertEqual(export.summary["postprocess_settings"]["near_fraction"], 0.07)

    def test_fraction_changes_counts_without_reanalysis(self):
        self.add_near_and_far()
        self.run_export()
        everyone = self.run_export("--near-fraction", "0")
        self.assertEqual((everyone.summary["cached_images"], everyone.summary["analyzed_images"]), (1, 0))
        self.assertEqual(everyone.summary["postprocess_settings"]["near_fraction"], 0.0)
        row = everyone.row("camA/IMG_0001.jpg")
        self.assertEqual((row["people_near"], row["people_near_class"]), ("2", "2"))
        self.assertEqual(everyone.events[0]["people_near_unique"], "2")
        nobody = self.run_export("--near-fraction", "0.5")   # 240 px
        row = nobody.row("camA/IMG_0001.jpg")
        self.assertEqual((row["people_total"], row["people_near"], row["people_near_class"]), ("2", "0", "0"))
        self.assertEqual((nobody.events[0]["people_near_max_frame"], nobody.events[0]["people_near_unique"]), ("0", "0"))

    def test_near_fraction_is_validated_before_any_work(self):
        between = "--near-fraction must be between 0 and 1."
        for value, message in (("-0.1", between), ("1", between), ("1.5", between), ("nan", between),
                               ("inf", between), ("abc", "invalid float value")):
            with self.subTest(value=value):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    self.args(f"--near-fraction={value}")
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
        self.assertEqual(self.args("--near-fraction", "0").near_fraction, 0.0)
        self.assertEqual(self.args("--near-fraction", "0.999").near_fraction, 0.999)
        # argparse converts string defaults with ``type``; other values are checked explicitly.
        for values, message in (({"near_fraction": "x"}, "invalid float value: 'x'"),
                                ({"near_fraction": [0.07]}, "--near-fraction must be a number."),
                                ({"near_fraction": 2}, between)):
            with self.subTest(values=values):
                (self.root / "settings.json").write_text(json.dumps(values), encoding="utf-8")
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                    self.args()
                self.assertIn(message, stderr.getvalue())
        (self.root / "settings.json").write_text(json.dumps({"near_fraction": 0.2}), encoding="utf-8")
        self.assertEqual(self.args().near_fraction, 0.2)
        self.ensure.assert_not_called()


class FakeAgeEngines:
    """Stands in for ``trailcam.vlm_age.OllamaAge`` (constructor signature ``(model, host, mode)``).

    ``photo`` is a list of whole-photo answers handed out in call order (a
    dict of counts, or an error string); ``ages`` maps a person's box left
    edge (rounded) to the age the fake model answers. ``engines`` and ``calls``
    record what ``__main__`` did.
    """

    def __init__(self, available=True, photo=(), ages=None):
        self.available_answer, self.photo, self.ages = available, list(photo), dict(ages or {})
        self.engines, self.calls = [], []

    def __call__(self, model, host="http://localhost:11434", mode="photo"):
        engine = SimpleNamespace(model=model, host=host, mode=mode)
        config = {"model": model, "mode": mode, "prompt_version": "vlm_age_v1", "max_people": 40}

        def available():
            self.calls.append(("available", mode))
            return self.available_answer

        def count_photo(image):
            self.calls.append(("count_photo", image.size, image.mode))
            answer = self.photo.pop(0)
            if isinstance(answer, str):
                return {"counts": None, "seconds": 0.01, "error": answer, **config}
            return {"counts": dict(answer), "seconds": 0.01, "error": None, **config}

        def classify(image, persons):
            self.calls.append(("classify", image.size, sorted(p["person_id"] for p in persons)))
            ages = {p["person_id"]: self.ages[round(p["xyxy"][0])] for p in persons if round(p["xyxy"][0]) in self.ages}
            return {"ages": ages, "seconds": 0.01, "requests": 1, "error": None, **config}

        engine.available, engine.count_photo, engine.classify = available, count_photo, classify
        self.engines.append(engine)
        return engine

    def installed(self):
        return patch("trailcam.vlm_age.OllamaAge", self)

    def asked(self, kind):
        return [call for call in self.calls if call[0] == kind]


def counts(adults=0, teens=0, children=0, unclear=0):
    return {"people_total": adults + teens + children + unclear, "adults": adults, "teens": teens,
            "children": children, "unclear": unclear}


class AgeModelTests(PipelineCase):
    """``--age-model`` / ``--age-mode`` / ``--ollama-host`` with a fake Ollama engine."""

    FAMILY = "camA/IMG_0001.jpg"
    VLM_COLUMNS = ("age_model", "vlm_adults", "vlm_teens", "vlm_children", "vlm_age_unclear")

    def add_family(self):
        """Geometry: A (x 60) adult, B (x 260, short) child, C (x 450) adult - all by relative height."""
        self.add(self.FAMILY, fakes.scene(fakes.walker(60), fakes.walker(260, y=170, height=130, color=CHILD_COLOR),
                                          fakes.walker(450, color=(40, 200, 40))), "2026:05:01 10:00:00")

    def vlm_cache(self, export, mode, relative, model="gemma4_26b"):
        """Changed: ``<sha256>_<16 hex>.json``, the suffix being sha256(fingerprint + people
        signature)[:16] for mosaic and sha256(fingerprint)[:16] for photo (was the person
        count). The signature is rebuilt here from the exported person ids and rounded boxes."""
        signature = ""
        if mode == "mosaic":
            people = json.loads(export.row(relative)["persons_json"])
            signature = hashlib.sha256(json.dumps(sorted((p["id"], p["box"]) for p in people)).encode()).hexdigest()[:16]
        suffix = hashlib.sha256((export.summary["configuration_hash"] + signature).encode()).hexdigest()[:16]
        return self.cache / "vlm_age" / f"{model}_{mode}_vlm_age_v1" / f"{file_hash(self.images / relative)}_{suffix}.json"

    def vlm_files(self, mode, model="gemma4_26b"):
        folder = self.cache / "vlm_age" / f"{model}_{mode}_vlm_age_v1"
        return sorted(path.name for path in folder.glob("*.json")) if folder.exists() else []

    @staticmethod
    def ages(row):
        return {p["box"][0]: (p["age"], p["age_method"]) for p in json.loads(row["persons_json"])}

    def test_age_model_is_off_by_default(self):
        self.add_family()
        fake = FakeAgeEngines()
        with fake.installed():
            export = self.run_export()
        self.assertEqual(fake.engines, [])
        row = export.row(self.FAMILY)
        self.assertEqual({k: row[k] for k in self.VLM_COLUMNS}, dict.fromkeys(self.VLM_COLUMNS, ""))
        self.assertEqual(self.ages(row), {60: ("unknown", "relative_height"), 260: ("child", "relative_height"),
                                          450: ("unknown", "relative_height")})
        self.assertFalse((self.cache / "vlm_age").exists())
        args = self.args()
        self.assertEqual((args.age_model, args.age_mode, args.ollama_host), (None, "photo", "http://localhost:11434"))

    def test_photo_mode_exports_counts_and_serves_them_from_the_cache(self):
        self.add_family()
        self.add("camA/IMG_0002.jpg", fakes.scene(fakes.walker(200)), "2026:05:01 10:00:10")
        fake = FakeAgeEngines(photo=[counts(adults=2, children=1), counts(adults=1, teens=1)])
        with fake.installed():
            export = self.run_export("--age-model", "gemma4:26b")
        engine, = fake.engines
        self.assertEqual((engine.model, engine.host, engine.mode), ("gemma4:26b", "http://localhost:11434", "photo"))
        self.assertEqual(fake.calls, [("available", "photo"), ("count_photo", (640, 480), "RGB"),
                                      ("count_photo", (640, 480), "RGB")])
        first, second = export.row(self.FAMILY), export.row("camA/IMG_0002.jpg")
        # Changed: age_by_vlm is blank (not 0) without per-person (mosaic) answers.
        self.assertEqual({k: first[k] for k in self.VLM_COLUMNS + ("age_by_vlm",)},
                         {"age_model": "gemma4:26b", "vlm_adults": "2", "vlm_teens": "0", "vlm_children": "1",
                          "vlm_age_unclear": "0", "age_by_vlm": ""})
        self.assertEqual((export.summary["age_model"], export.summary["age_mode"],
                          export.summary["age_model_failures"]), ("gemma4:26b", "photo", 0))
        self.assertEqual((second["vlm_adults"], second["vlm_teens"], second["vlm_children"]), ("1", "1", "0"))
        # Whole-photo counts never change the per-person (geometry) ages.
        self.assertEqual(self.ages(first), {60: ("unknown", "relative_height"), 260: ("child", "relative_height"),
                                            450: ("unknown", "relative_height")})
        event = export.event_of(self.FAMILY)
        self.assertEqual(event["image_count"], "2")
        self.assertEqual({k: event[f"vlm_{k}_max_frame"] for k in ("adults", "teens", "children", "unclear")},
                         {"adults": "2", "teens": "1", "children": "1", "unclear": "0"})
        cached = json.loads(self.vlm_cache(export, "photo", self.FAMILY).read_text(encoding="utf-8"))
        self.assertEqual((cached["counts"], cached["error"]), (counts(adults=2, children=1), None))
        self.assertTrue(self.vlm_cache(export, "photo", "camA/IMG_0002.jpg").exists())
        self.assertEqual(len(self.vlm_files("photo")), 2)

        again = FakeAgeEngines()   # would fail (no scripted answers) if asked
        with again.installed():
            second_run = self.run_export("--age-model", "gemma4:26b")
        self.assertEqual(again.calls, [("available", "photo")])
        self.assertEqual(second_run.summary["cached_images"], 2)
        for relative in (self.FAMILY, "camA/IMG_0002.jpg"):
            self.assertEqual(without(second_run.row(relative)), without(export.row(relative)))
        self.assertEqual([without(e, ("run_id",)) for e in second_run.events],
                         [without(e, ("run_id",)) for e in export.events])

    def test_mosaic_mode_replaces_geometry_ages(self):
        self.add_family()
        fake = FakeAgeEngines(ages={60: "child", 260: "teen", 450: "unclear"})
        with fake.installed():
            export = self.run_export("--age-model", "gemma4:26b", "--age-mode", "mosaic")
        self.assertEqual([e.mode for e in fake.engines], ["mosaic"])
        (_, size, ids), = fake.asked("classify")
        self.assertEqual((size, len(ids)), ((640, 480), 3))
        self.assertEqual(fake.asked("count_photo"), [])
        row = export.row(self.FAMILY)
        # A: adult -> child (VLM); B: child -> teen -> unknown; C: "unclear" keeps the geometry adult.
        self.assertEqual(self.ages(row), {60: ("child", "vlm"), 260: ("unknown", "vlm"), 450: ("unknown", "relative_height")})
        self.assertEqual((row["adults"], row["children"], row["age_unknown"]), ("0", "1", "2"))
        self.assertEqual((row["age_by_vlm"], row["age_by_relative_height"], row["age_model"]), ("1", "0", "gemma4:26b"))
        self.assertEqual((row["vlm_adults"], row["vlm_children"]), ("", ""))
        event = export.event_of(self.FAMILY)
        self.assertEqual((event["adults"], event["children"], event["age_unknown"]), ("0", "1", "2"))
        self.assertEqual(export.summary["age_mode"], "mosaic")
        cached = json.loads(self.vlm_cache(export, "mosaic", self.FAMILY).read_text(encoding="utf-8"))
        self.assertEqual(sorted(cached["ages"].values()), ["child", "teen", "unclear"])
        self.assertEqual(set(cached["ages"]), set(ids))
        self.assertEqual(self.vlm_files("photo"), [])

        again = FakeAgeEngines()
        with again.installed():
            second_run = self.run_export("--age-model", "gemma4:26b", "--age-mode", "mosaic")
        self.assertEqual(again.calls, [("available", "mosaic")])
        self.assertEqual(without(second_run.row(self.FAMILY)), without(row))

    def test_photo_plus_mosaic_asks_both_and_caches_each(self):
        self.add_family()
        fake = FakeAgeEngines(photo=[counts(adults=1, children=2)], ages={450: "child"})
        with fake.installed():
            export = self.run_export("--age-model", "gemma4:26b", "--age-mode", "photo+mosaic",
                                     "--ollama-host", "http://gpu-box:11434")
        self.assertEqual([(e.model, e.host, e.mode) for e in fake.engines],
                         [("gemma4:26b", "http://gpu-box:11434", "photo"), ("gemma4:26b", "http://gpu-box:11434", "mosaic")])
        self.assertEqual([c[0] for c in fake.calls], ["available", "count_photo", "classify"])
        self.assertEqual(fake.asked("available"), [("available", "photo")])   # one check for both modes
        row = export.row(self.FAMILY)
        self.assertEqual((row["vlm_adults"], row["vlm_children"], row["age_by_vlm"]), ("1", "2", "1"))
        self.assertEqual((row["adults"], row["children"]), ("0", "2"))
        self.assertTrue(self.vlm_cache(export, "photo", self.FAMILY).exists())
        self.assertTrue(self.vlm_cache(export, "mosaic", self.FAMILY).exists())
        # Photo and mosaic keys differ (the mosaic key also covers the people asked about).
        self.assertNotEqual(self.vlm_files("photo"), self.vlm_files("mosaic"))

    def test_unavailable_server_falls_back_to_geometry_with_a_message(self):
        self.add_family()
        fake = FakeAgeEngines(available=False)
        with fake.installed():
            export = self.run_export("--age-model", "gemma4:26b", "--ollama-host", "http://127.0.0.1:9")
        # Changed: the run continues offline (cached answers where present) instead of
        # dropping the age expert, and says so.
        self.assertIn("Age model gemma4:26b is not available at http://127.0.0.1:9; using cached answers where "
                      "present and geometry otherwise.", self.stderr.getvalue())
        self.assertEqual(fake.calls, [("available", "photo")])
        row = export.row(self.FAMILY)
        self.assertEqual({k: row[k] for k in self.VLM_COLUMNS + ("age_by_vlm",)},
                         dict.fromkeys(self.VLM_COLUMNS + ("age_by_vlm",), ""))
        self.assertEqual(self.ages(row)[260], ("child", "relative_height"))
        self.assertFalse((self.cache / "vlm_age").exists())
        self.assertEqual((export.summary["age_model"], export.summary["age_mode"],
                          export.summary["age_model_failures"]), ("gemma4:26b", "photo", 0))

    def test_real_engine_without_a_server_falls_back_to_geometry(self):
        self.add_family()

        def refuse(*args, **kwargs):
            raise OSError("connection refused")

        def never(*args, **kwargs):
            raise AssertionError("no request may be sent when the server is down")

        requests = types.ModuleType("requests")
        requests.get, requests.post = refuse, never
        with patch.dict(sys.modules, {"requests": requests}):
            export = self.run_export("--age-model", "gemma4:26b", "--age-mode", "photo+mosaic")
        self.assertIn("Age model gemma4:26b is not available at http://localhost:11434; using cached answers where "
                      "present and geometry otherwise.", self.stderr.getvalue())
        self.assertEqual(export.row(self.FAMILY)["age_model"], "")
        self.assertEqual(export.summary["age_model_failures"], 0)

    def test_invalid_age_mode_is_rejected_before_any_work(self):
        self.add_family()
        fake = FakeAgeEngines()
        with fake.installed():
            for mode in ("marks", "crops", "mosaic+photo", "Photo"):
                with self.subTest(mode=mode):
                    stderr = io.StringIO()
                    with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                        self.args("--age-model", "gemma4:26b", "--age-mode", mode)
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn("invalid choice", stderr.getvalue())
            (self.root / "settings.json").write_text(json.dumps({"age_mode": "crops"}), encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                entry.main(["--input", str(self.images), "--output", str(self.output), "--models", str(self.models),
                            "--cache", str(self.cache), "--age-model", "gemma4:26b"])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("Invalid age_mode 'crops'", stderr.getvalue())
        self.assertEqual(fake.engines, [])
        self.ensure.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_age_settings_from_the_settings_file(self):
        (self.root / "settings.json").write_text(json.dumps({
            "age_model": "gemma4:26b", "age_mode": "photo+mosaic", "ollama_host": "http://gpu-box:11434"}),
            encoding="utf-8")
        args = self.args()
        self.assertEqual((args.age_model, args.age_mode, args.ollama_host),
                         ("gemma4:26b", "photo+mosaic", "http://gpu-box:11434"))
        args = self.args("--age-mode", "photo", "--age-model", "qwen2.5vl:7b")
        self.assertEqual((args.age_model, args.age_mode), ("qwen2.5vl:7b", "photo"))

    def test_failed_answers_are_not_cached_and_are_asked_again(self):
        self.add_family()
        fake = FakeAgeEngines(photo=["ConnectionError: connection reset"])
        with fake.installed():
            export = self.run_export("--age-model", "gemma4:26b")
        row = export.row(self.FAMILY)
        # Changed: age_model names the model only when it actually answered for the photo.
        self.assertEqual(row["age_model"], "")
        self.assertEqual((row["vlm_adults"], row["vlm_children"]), ("", ""))
        self.assertEqual(export.event_of(self.FAMILY)["vlm_children_max_frame"], "")
        self.assertFalse(self.vlm_cache(export, "photo", self.FAMILY).exists())
        self.assertEqual(export.summary["age_model_failures"], 1)
        self.assertNotIn("in a row", self.stderr.getvalue())   # one failure does not switch the model off
        retry = FakeAgeEngines(photo=[counts(adults=2, children=1)])
        with retry.installed():
            export = self.run_export("--age-model", "gemma4:26b")
        self.assertEqual(len(retry.asked("count_photo")), 1)
        self.assertEqual(export.row(self.FAMILY)["vlm_children"], "1")
        self.assertEqual(export.row(self.FAMILY)["age_model"], "gemma4:26b")
        self.assertTrue(self.vlm_cache(export, "photo", self.FAMILY).exists())
        self.assertEqual(export.summary["age_model_failures"], 0)

    def test_mosaic_skips_photos_without_people(self):
        # Changed: with the empty-frame gate on, an empty photo is gated_empty and photo
        # mode skips it too (see test_gated_empty_frame_is_not_sent_to_the_model); the gate
        # is off here so the whole photo is still counted.
        self.add("camA/IMG_0005.jpg", fakes.scene(), "2026:05:01 10:00:00")
        fake = FakeAgeEngines(photo=[counts()])
        with fake.installed():
            export = self.run_export("--age-model", "gemma4:26b", "--age-mode", "photo+mosaic",
                                     "--no-empty-frame-gate")
        self.assertEqual(export.row("camA/IMG_0005.jpg")["gated_empty"], "False")
        self.assertEqual(fake.asked("classify"), [])
        self.assertEqual(len(fake.asked("count_photo")), 1)   # the whole photo is still counted
        self.assertEqual(self.vlm_files("mosaic"), [])         # nothing asked, nothing cached
        self.assertTrue(self.vlm_cache(export, "photo", "camA/IMG_0005.jpg").exists())
        row = export.row("camA/IMG_0005.jpg")
        # No people: the mosaic counts as answered with no ages, so age_by_vlm is 0, not blank.
        self.assertEqual((row["people_total"], row["vlm_adults"], row["age_by_vlm"], row["age_model"]),
                         ("0", "0", "0", "gemma4:26b"))

    def test_gated_empty_frame_is_not_sent_to_the_model(self):
        self.add_family()
        self.add("camA/IMG_0005.jpg", fakes.scene(), "2026:05:01 10:05:00")   # the gate closes: gated_empty
        fake = FakeAgeEngines(photo=[counts(adults=2, children=1)])
        with fake.installed():
            export = self.run_export("--age-model", "gemma4:26b")
        empty = export.row("camA/IMG_0005.jpg")
        self.assertEqual(empty["gated_empty"], "True")
        self.assertEqual(fake.asked("count_photo"), [("count_photo", (640, 480), "RGB")])   # the family photo only
        self.assertEqual({k: empty[k] for k in self.VLM_COLUMNS + ("age_by_vlm",)},
                         dict.fromkeys(self.VLM_COLUMNS + ("age_by_vlm",), ""))
        self.assertEqual(export.row(self.FAMILY)["vlm_children"], "1")
        self.assertEqual(self.vlm_files("photo"), [self.vlm_cache(export, "photo", self.FAMILY).name])
        self.assertEqual(export.summary["age_model_failures"], 0)
        # photo+mosaic: still no request for the gated frame; the (empty) mosaic answer is trivial.
        both = FakeAgeEngines(ages={60: "child"})
        with both.installed():
            export = self.run_export("--age-model", "gemma4:26b", "--age-mode", "photo+mosaic")
        self.assertEqual([c[0] for c in both.calls], ["available", "classify"])   # family photo served from cache
        empty = export.row("camA/IMG_0005.jpg")
        self.assertEqual((empty["vlm_adults"], empty["age_by_vlm"], empty["age_model"]), ("", "0", "gemma4:26b"))

    def test_cache_folder_name_is_safe_for_any_model_name(self):
        self.add_family()
        fake = FakeAgeEngines(photo=[counts(adults=3)])
        with fake.installed():
            export = self.run_export("--age-model", "library/gemma4:26b")
        self.assertTrue(self.vlm_cache(export, "photo", self.FAMILY, model="library_gemma4_26b").exists())
        self.assertEqual([p.name for p in (self.cache / "vlm_age").iterdir()], ["library_gemma4_26b_photo_vlm_age_v1"])

    def test_the_model_sees_the_exif_oriented_photo(self):
        rotated = self.images / "camA" / "IMG_0009.jpg"
        rotated.parent.mkdir(parents=True, exist_ok=True)
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90 degrees on display
        Image.new("RGB", (640, 480), (90, 110, 70)).save(rotated, "JPEG", exif=exif.tobytes())
        fake = FakeAgeEngines(photo=[counts()])
        with fake.installed():
            # Changed: an empty (gated) frame is no longer sent, so the gate is off here.
            self.run_export("--age-model", "gemma4:26b", "--no-empty-frame-gate")
        self.assertEqual(fake.asked("count_photo"), [("count_photo", (480, 640), "RGB")])

    def test_unreadable_image_with_an_age_model_is_exported_as_an_error(self):
        # Regression guard: prepare() now always sets item["prepared"] (False for
        # records that are not ok and for fusion failures) and run() only asks the
        # age expert about prepared images; a missing key used to abort the run.
        self.add_family()
        (self.images / "camA" / "broken.jpg").write_bytes(b"not a jpeg")
        self.add("camA/bad.jpg", fakes.scene(fakes.walker(300), corrupt_postprocess=True), "2026:05:01 10:00:05")
        fake = FakeAgeEngines(photo=[counts(adults=2, children=1)])
        with fake.installed():
            export = self.run_export("--age-model", "gemma4:26b", expect=1)
        self.assertEqual(export.row("camA/broken.jpg")["status"], "error")
        self.assertEqual(export.row("camA/bad.jpg")["error"], "Post-processing failed: fusion: KeyError: 'detection_index'")
        self.assertEqual(len(fake.asked("count_photo")), 1)   # only the prepared family photo is asked
        self.assertEqual((export.row("camA/bad.jpg")["age_model"], export.row(self.FAMILY)["age_model"]),
                         ("", "gemma4:26b"))

    def test_cached_person_answers_survive_a_reanalysis(self):
        # Regression guard: person ids are re-assigned (by confidence) when an image is
        # re-analysed with another configuration (--profile, models) or --force. The
        # mosaic cache key now covers the inference fingerprint and the exact people
        # asked about (ids and rounded boxes), so a re-analysis asks again instead of
        # applying cached ages to other people. (Was keyed by <sha>_<person count>.)
        busy = {"yoloe": .9, "yolo26n": .5, "megadetector": .5}
        quiet = {"yoloe": .6, "yolo26n": .5, "megadetector": .5}
        self.add("camA/IMG_0001.jpg", fakes.scene(fakes.walker(60, experts=busy),
                                                  fakes.walker(300, experts=quiet, color=(40, 200, 40))),
                 "2026:05:01 10:00:00")
        fake = FakeAgeEngines(ages={60: "child", 300: "adult"})
        with fake.installed():
            first = self.run_export("--age-model", "gemma4:26b", "--age-mode", "mosaic")
            self.assertEqual(self.ages(first.row("camA/IMG_0001.jpg"))[60], ("child", "vlm"))
            self.scenes["camA/IMG_0001.jpg"] = fakes.scene(fakes.walker(60, experts=quiet),
                                                           fakes.walker(300, experts=busy, color=(40, 200, 40)))
            second = self.run_export("--age-model", "gemma4:26b", "--age-mode", "mosaic", "--profile", "fast")
        self.assertEqual(second.summary["analyzed_images"], 1)
        self.assertEqual(self.ages(second.row("camA/IMG_0001.jpg"))[60], ("child", "vlm"))
        self.assertEqual(self.ages(second.row("camA/IMG_0001.jpg"))[300], ("adult", "vlm"))
        self.assertEqual(len(fake.asked("classify")), 2)   # asked again for the re-analysed people
        self.assertEqual(len(self.vlm_files("mosaic")), 2)

    def test_offline_run_reuses_cached_answers_and_keeps_geometry_elsewhere(self):
        self.add_family()
        fake = FakeAgeEngines(photo=[counts(adults=2, children=1)], ages={60: "child", 260: "teen", 450: "unclear"})
        with fake.installed():
            online = self.run_export("--age-model", "gemma4:26b", "--age-mode", "photo+mosaic")
        cached_files = (self.vlm_files("photo"), self.vlm_files("mosaic"))
        self.add("camB/IMG_0002.jpg", fakes.scene(fakes.walker(200)), "2026:05:02 10:00:00")   # never answered
        down = FakeAgeEngines(available=False)
        with down.installed():
            offline = self.run_export("--age-model", "gemma4:26b", "--age-mode", "photo+mosaic")
        self.assertIn("Age model gemma4:26b is not available at http://localhost:11434; using cached answers where "
                      "present and geometry otherwise.", self.stderr.getvalue())
        self.assertEqual(down.calls, [("available", "photo")])   # no request of any kind
        family = offline.row(self.FAMILY)
        self.assertEqual(without(family), without(online.row(self.FAMILY)))
        self.assertEqual((family["vlm_adults"], family["vlm_children"], family["age_by_vlm"], family["age_model"]),
                         ("2", "1", "1", "gemma4:26b"))
        self.assertEqual(self.ages(family)[60], ("child", "vlm"))
        new = offline.row("camB/IMG_0002.jpg")
        self.assertEqual({k: new[k] for k in self.VLM_COLUMNS + ("age_by_vlm",)},
                         dict.fromkeys(self.VLM_COLUMNS + ("age_by_vlm",), ""))
        self.assertNotIn("vlm", {method for _, method in self.ages(new).values()})
        self.assertEqual((self.vlm_files("photo"), self.vlm_files("mosaic")), cached_files)
        self.assertEqual((offline.summary["age_model"], offline.summary["age_mode"],
                          offline.summary["age_model_failures"]), ("gemma4:26b", "photo+mosaic", 0))

    def add_series(self, frames=4):
        for n in range(1, frames + 1):
            self.add(f"camA/IMG_000{n}.jpg", fakes.scene(fakes.walker(40 + 60 * n)), f"2026:05:01 10:00:{10 * n:02d}")

    def test_three_consecutive_failures_switch_to_offline(self):
        self.add("camA/IMG_0009.jpg", fakes.scene(fakes.walker(300)), "2026:05:01 11:00:00")
        seed = FakeAgeEngines(photo=[counts(adults=1)])
        with seed.installed():
            self.run_export("--age-model", "gemma4:26b")
        self.add_series(4)   # IMG_0001..0004 come before the cached IMG_0009
        failing = FakeAgeEngines(photo=["ConnectionError: reset"] * 3)
        with failing.installed():
            export = self.run_export("--age-model", "gemma4:26b")
        self.assertEqual(len(failing.asked("count_photo")), 3)   # IMG_0004 is not asked any more
        message = ("Age model failed 3 times in a row; continuing with cached answers and geometry only. "
                   "Last error: photo: ConnectionError: reset")
        self.assertEqual(self.stderr.getvalue().count(message), 1)
        self.assertEqual(export.summary["age_model_failures"], 3)
        for n in range(1, 5):
            with self.subTest(frame=n):
                row = export.row(f"camA/IMG_000{n}.jpg")
                self.assertEqual((row["age_model"], row["vlm_adults"]), ("", ""))
        cached = export.row("camA/IMG_0009.jpg")   # after the switch cached answers are still used
        self.assertEqual((cached["age_model"], cached["vlm_adults"]), ("gemma4:26b", "1"))
        self.assertEqual(len(self.vlm_files("photo")), 1)   # failures are never cached

    def test_a_valid_answer_resets_the_failure_streak(self):
        self.add_series(4)
        self.add("camA/IMG_0005.jpg", fakes.scene(fakes.walker(400)), "2026:05:01 10:00:50")
        fake = FakeAgeEngines(photo=["ValueError: x", "ValueError: x", counts(adults=1), "ValueError: x",
                                     "ValueError: x"])
        with fake.installed():
            export = self.run_export("--age-model", "gemma4:26b")
        self.assertEqual(len(fake.asked("count_photo")), 5)
        self.assertNotIn("in a row", self.stderr.getvalue())
        self.assertEqual(export.summary["age_model_failures"], 4)
        self.assertEqual([export.row(f"camA/IMG_000{n}.jpg")["vlm_adults"] for n in range(1, 6)], ["", "", "1", "", ""])

    def test_failures_of_either_mode_count_towards_the_streak(self):
        # photo+mosaic: each mode's failed request counts; the third switches off within the photo.
        self.add_series(2)
        fake = FakeAgeEngines(photo=["ValueError: x", "ValueError: x"])
        failing_classify = []

        def wrap(engine):
            def classify(image, persons):
                failing_classify.append(sorted(p["person_id"] for p in persons))
                return {"ages": {}, "seconds": 0.0, "requests": 0, "error": "ConnectionError: reset"}
            engine.classify = classify
            return engine

        with patch("trailcam.vlm_age.OllamaAge", lambda *a, **k: wrap(fake(*a, **k))):
            export = self.run_export("--age-model", "gemma4:26b", "--age-mode", "photo+mosaic")
        # frame 1: photo fails (1), mosaic fails (2); frame 2: photo fails (3) -> offline, mosaic not asked.
        self.assertEqual((len(fake.asked("count_photo")), len(failing_classify)), (2, 1))
        self.assertEqual(export.summary["age_model_failures"], 3)
        self.assertIn("Last error: photo: ValueError: x", self.stderr.getvalue())
        self.assertEqual(self.vlm_files("mosaic"), [])

    def test_force_asks_the_model_again_and_replaces_the_cached_answer(self):
        self.add_family()
        first = FakeAgeEngines(photo=[counts(adults=2, children=1)], ages={60: "child"})
        with first.installed():
            export = self.run_export("--age-model", "gemma4:26b", "--age-mode", "photo+mosaic")
        again = FakeAgeEngines(photo=[counts(adults=3)], ages={60: "adult"})
        with again.installed():
            forced = self.run_export("--age-model", "gemma4:26b", "--age-mode", "photo+mosaic", "--force")
        self.assertEqual([c[0] for c in again.calls], ["available", "count_photo", "classify"])
        row = forced.row(self.FAMILY)
        self.assertEqual((row["vlm_adults"], row["vlm_children"]), ("3", "0"))
        self.assertEqual(self.ages(row)[60], ("adult", "vlm"))
        photo = json.loads(self.vlm_cache(forced, "photo", self.FAMILY).read_text(encoding="utf-8"))
        self.assertEqual(photo["counts"], counts(adults=3))
        self.assertEqual((len(self.vlm_files("photo")), len(self.vlm_files("mosaic"))), (1, 1))   # replaced in place
        quiet = FakeAgeEngines()
        with quiet.installed():
            later = self.run_export("--age-model", "gemma4:26b", "--age-mode", "photo+mosaic")
        self.assertEqual(quiet.calls, [("available", "photo")])
        self.assertEqual((later.row(self.FAMILY)["vlm_adults"], self.ages(later.row(self.FAMILY))[60]),
                         ("3", ("adult", "vlm")))

    def test_invalid_cached_answers_are_asked_again(self):
        self.add_family()
        seed = FakeAgeEngines(photo=[counts(adults=2, children=1)], ages={60: "child"})
        with seed.installed():
            export = self.run_export("--age-model", "gemma4:26b", "--age-mode", "photo+mosaic")
        photo_file, mosaic_file = self.vlm_cache(export, "photo", self.FAMILY), self.vlm_cache(export, "mosaic", self.FAMILY)
        photo, mosaic = (json.loads(p.read_text(encoding="utf-8")) for p in (photo_file, mosaic_file))
        some_id = next(iter(mosaic["ages"]))

        def with_counts(**change):
            return json.dumps({**photo, "counts": {**photo["counts"], **change}})

        teenless = {k: v for k, v in photo["counts"].items() if k != "teens"}
        broken_photo = {"not_json": "{not json", "json_list": json.dumps([photo]), "json_null": "null",
                        "error_answer": json.dumps({**photo, "counts": None, "error": "ValueError: x"}),
                        "error_with_counts": json.dumps({**photo, "error": "ConnectionError: reset"}),
                        "counts_missing_key": json.dumps({**photo, "counts": teenless}),
                        "negative": with_counts(adults=-1), "text": with_counts(adults="2"),
                        "float": with_counts(adults=2.0), "bool": with_counts(children=True),
                        "counts_list": json.dumps({**photo, "counts": [2, 0, 1, 0]})}
        broken_mosaic = {"not_json": "{", "no_ages": json.dumps({"error": None}),
                         "unknown_age": json.dumps({**mosaic, "ages": {some_id: "elderly"}}),
                         "null_age": json.dumps({**mosaic, "ages": {some_id: None}}),
                         "ages_list": json.dumps({**mosaic, "ages": ["child"]}),
                         "error_answer": json.dumps({**mosaic, "error": "ConnectionError: reset"})}
        for mode, target, variants in (("photo", photo_file, broken_photo), ("mosaic", mosaic_file, broken_mosaic)):
            for name, text in variants.items():
                with self.subTest(mode=mode, variant=name):
                    target.write_text(text, encoding="utf-8")
                    fake = FakeAgeEngines(photo=[counts(adults=1, unclear=2)], ages={60: "adult"})
                    with fake.installed():
                        run = self.run_export("--age-model", "gemma4:26b", "--age-mode", "photo+mosaic")
                    asked = [c[0] for c in fake.calls if c[0] != "available"]
                    self.assertEqual(asked, ["count_photo" if mode == "photo" else "classify"])
                    row = run.row(self.FAMILY)
                    if mode == "photo":
                        self.assertEqual((row["vlm_adults"], row["vlm_age_unclear"]), ("1", "2"))
                        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["counts"],
                                         counts(adults=1, unclear=2))
                    else:
                        self.assertEqual(self.ages(row)[60], ("adult", "vlm"))
                        self.assertEqual(set(json.loads(target.read_text(encoding="utf-8"))["ages"].values()), {"adult"})
                    self.assertEqual(run.summary["age_model_failures"], 0)   # a bad cache file is not a model failure


class AddVlmAgeTests(unittest.TestCase):
    """``__main__.add_vlm_age`` on hand-built prepared items: cache keys, validation,
    offline mode and the failure streak, without the pipeline around it."""

    SHA = "ab" * 32
    FINGERPRINT = "f" * 64
    PEOPLE = (("cand_0002", [300.4, 50.0, 340.6, 150.0]), ("cand_0001", [10.2, 20.0, 110.0, 320.0]))
    # sorted((person_id, rounded xyxy)) as JSON: the documented people signature input.
    SIGNATURE_TEXT = '[["cand_0001", [10, 20, 110, 320]], ["cand_0002", [300, 50, 341, 150]]]'

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.photo = self.root / "photo.jpg"
        Image.new("RGB", (640, 480), (90, 110, 70)).save(self.photo, "JPEG")
        self.args = SimpleNamespace(age_model="gemma4:26b", cache=self.root / "cache", force=False)
        self.calls, self.replies = [], {"photo": [], "mosaic": []}
        self.stderr = io.StringIO()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(contextlib.redirect_stderr(self.stderr))

    def reply(self, mode):
        answer = self.replies[mode].pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def state(self, modes=("photo", "mosaic"), online=True):
        def count_photo(image):
            self.calls.append(("photo", image.size))
            return self.reply("photo")

        def classify(image, persons):
            self.calls.append(("mosaic", sorted(p["person_id"] for p in persons)))
            return self.reply("mosaic")

        engine = SimpleNamespace(count_photo=count_photo, classify=classify)
        return {"engines": {mode: engine for mode in modes}, "online": online, "failures": 0, "consecutive": 0,
                "fingerprint": None}

    def item(self, people=PEOPLE, gated=False, sha=SHA):
        return {"sha256": sha, "result": {"persons": [{"person_id": pid, "xyxy": list(box)} for pid, box in people],
                                          "gated_empty": gated}}

    @staticmethod
    def photo_answer(adults=1, children=0):
        return {"counts": {"people_total": adults + children, "adults": adults, "teens": 0, "children": children,
                           "unclear": 0}, "seconds": 0.1, "error": None}

    @staticmethod
    def mosaic_answer(**ages):
        return {"ages": dict(ages), "seconds": 0.1, "requests": 1, "error": None}

    def expected(self, mode, people=PEOPLE, fingerprint=FINGERPRINT, sha=SHA):
        signature = ""
        if mode == "mosaic":
            text = json.dumps(sorted((pid, [round(v) for v in box]) for pid, box in people))
            signature = hashlib.sha256(text.encode()).hexdigest()[:16]
        suffix = hashlib.sha256((fingerprint + signature).encode()).hexdigest()[:16]
        return self.args.cache / "vlm_age" / f"gemma4_26b_{mode}_vlm_age_v1" / f"{sha}_{suffix}.json"

    def cached_files(self):
        folder = self.args.cache / "vlm_age"
        return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*.json")) if folder.exists() else []

    def ask(self, state, item, fingerprint=FINGERPRINT, path=None):
        entry.add_vlm_age(state, self.args, item, path or self.photo, fingerprint)
        return item["result"]["vlm_age"]

    def test_partial_mosaic_answer_is_used_but_not_cached(self):
        # A crowded photo whose later 16-person batch failed still gets the ages
        # that were answered; nothing is cached, so the photo is asked again.
        partial = {"ages": {"cand_0001": "child"}, "seconds": 0.1, "requests": 2,
                   "error": "ConnectionError: reset"}
        self.replies.update(mosaic=[partial, self.mosaic_answer(cand_0001="child", cand_0002="adult")])
        state = self.state(modes=("mosaic",))
        answer = self.ask(state, self.item())
        self.assertEqual(answer["model"], "gemma4:26b")
        self.assertEqual(answer["persons"]["ages"], {"cand_0001": "child"})
        self.assertEqual(answer["errors"], ["mosaic: partial answer: ConnectionError: reset"])
        self.assertEqual(self.cached_files(), [])
        self.assertEqual((state["failures"], state["online"]), (1, True))
        again = self.ask(state, self.item())
        self.assertEqual(again["persons"]["ages"], {"cand_0001": "child", "cand_0002": "adult"})
        self.assertEqual(len(self.cached_files()), 1)

    def test_cache_file_names(self):
        self.assertEqual(json.dumps(sorted((pid, [round(v) for v in box]) for pid, box in self.PEOPLE)),
                         self.SIGNATURE_TEXT)
        signature = hashlib.sha256(self.SIGNATURE_TEXT.encode()).hexdigest()[:16]
        mosaic_suffix = hashlib.sha256((self.FINGERPRINT + signature).encode()).hexdigest()[:16]
        photo_suffix = hashlib.sha256(self.FINGERPRINT.encode()).hexdigest()[:16]
        self.replies.update(photo=[self.photo_answer(2)], mosaic=[self.mosaic_answer(cand_0001="child")])
        answer = self.ask(self.state(), self.item())
        self.assertEqual(self.cached_files(), [f"gemma4_26b_mosaic_vlm_age_v1/{self.SHA}_{mosaic_suffix}.json",
                                               f"gemma4_26b_photo_vlm_age_v1/{self.SHA}_{photo_suffix}.json"])
        for suffix in (mosaic_suffix, photo_suffix):
            self.assertRegex(suffix, r"^[0-9a-f]{16}$")
        self.assertEqual(answer, {"model": "gemma4:26b", "photo": self.photo_answer(2),
                                  "persons": self.mosaic_answer(cand_0001="child"), "errors": None})
        self.assertEqual(json.loads(self.expected("photo").read_text(encoding="utf-8")), self.photo_answer(2))
        self.assertEqual(self.calls, [("photo", (640, 480)), ("mosaic", ["cand_0001", "cand_0002"])])

    def test_mosaic_key_changes_when_the_people_change(self):
        self.replies.update(photo=[self.photo_answer()], mosaic=[self.mosaic_answer(cand_0001="child")])
        self.ask(self.state(), self.item())
        self.assertEqual(len(self.calls), 2)
        # Same people (sub-pixel jitter rounds away): served from the cache.
        jitter = (("cand_0002", [300.3, 50.4, 340.51, 149.6]), ("cand_0001", [10.4, 19.8, 110.2, 320.0]))
        for people in (self.PEOPLE, tuple(reversed(self.PEOPLE)), jitter):
            with self.subTest(people=people):
                answer = self.ask(self.state(), self.item(people))
                self.assertEqual(answer["persons"], self.mosaic_answer(cand_0001="child"))
                self.assertEqual(len(self.calls), 2)
        # Different people: a new key, asked again; the photo answer does not depend on them.
        swapped = (("cand_0001", self.PEOPLE[0][1]), ("cand_0002", self.PEOPLE[1][1]))   # ids re-assigned
        moved = (self.PEOPLE[0], ("cand_0001", [10.2, 20.0, 180.0, 320.0]))
        fewer = (self.PEOPLE[1],)
        more = self.PEOPLE + (("cand_0003", [500.0, 50.0, 540.0, 150.0]),)
        for name, people in (("swapped", swapped), ("moved", moved), ("fewer", fewer), ("more", more)):
            with self.subTest(name):
                before = len(self.calls)
                self.replies["mosaic"].append(self.mosaic_answer(cand_0002="adult"))
                answer = self.ask(self.state(), self.item(people))
                self.assertEqual(self.calls[before:], [("mosaic", sorted(pid for pid, _ in people))])
                self.assertEqual(answer["persons"], self.mosaic_answer(cand_0002="adult"))
                self.assertTrue(self.expected("mosaic", people).exists())
        self.assertEqual(len([c for c in self.calls if c[0] == "photo"]), 1)
        self.assertEqual(len([f for f in self.cached_files() if "_mosaic_" in f]), 5)

    def test_keys_follow_the_inference_fingerprint(self):
        self.replies.update(photo=[self.photo_answer(1), self.photo_answer(2)],
                            mosaic=[self.mosaic_answer(cand_0001="child"), self.mosaic_answer(cand_0001="adult")])
        self.ask(self.state(), self.item())
        answer = self.ask(self.state(), self.item(), fingerprint="0" * 64)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual((answer["photo"], answer["persons"]), (self.photo_answer(2), self.mosaic_answer(cand_0001="adult")))
        self.assertEqual(len(self.cached_files()), 4)
        self.assertTrue(self.expected("photo", fingerprint="0" * 64).exists())

    def test_force_skips_the_cache_read(self):
        self.replies.update(photo=[self.photo_answer(1), self.photo_answer(3)],
                            mosaic=[self.mosaic_answer(cand_0001="child"), self.mosaic_answer(cand_0001="adult")])
        self.ask(self.state(), self.item())
        self.args.force = True
        answer = self.ask(self.state(), self.item())
        self.assertEqual(len(self.calls), 4)
        self.assertEqual((answer["photo"], answer["persons"]), (self.photo_answer(3), self.mosaic_answer(cand_0001="adult")))
        self.assertEqual(json.loads(self.expected("photo").read_text(encoding="utf-8")), self.photo_answer(3))
        self.assertEqual(len(self.cached_files()), 2)
        # Forced but offline: no cache read and no request, so nothing is attached.
        answer = self.ask(self.state(online=False), self.item())
        self.assertEqual(answer, {"model": None, "photo": None, "persons": None, "errors": None})
        self.assertEqual(len(self.calls), 4)

    def test_offline_state_uses_the_cache_only(self):
        self.replies.update(photo=[self.photo_answer(2)], mosaic=[self.mosaic_answer(cand_0001="child")])
        self.ask(self.state(), self.item())
        state = self.state(online=False)
        answer = self.ask(state, self.item())
        self.assertEqual(answer, {"model": "gemma4:26b", "photo": self.photo_answer(2),
                                  "persons": self.mosaic_answer(cand_0001="child"), "errors": None})
        other = self.item(sha="cd" * 32)
        self.assertEqual(self.ask(state, other), {"model": None, "photo": None, "persons": None, "errors": None})
        self.assertEqual(len(self.calls), 2)
        self.assertEqual((state["failures"], state["consecutive"], state["online"]), (0, 0, False))

    def test_invalid_cached_answer_is_asked_again(self):
        self.expected("mosaic").parent.mkdir(parents=True)
        self.expected("mosaic").write_text(json.dumps(self.mosaic_answer(cand_0001="elderly")), encoding="utf-8")
        self.expected("photo").parent.mkdir(parents=True)
        self.expected("photo").write_text(json.dumps({"counts": None, "error": "ValueError: x"}), encoding="utf-8")
        self.replies.update(photo=[self.photo_answer(1)], mosaic=[self.mosaic_answer(cand_0001="teen")])
        answer = self.ask(self.state(), self.item())
        self.assertEqual(len(self.calls), 2)
        self.assertEqual((answer["photo"], answer["persons"]), (self.photo_answer(1), self.mosaic_answer(cand_0001="teen")))
        self.assertEqual(json.loads(self.expected("mosaic").read_text(encoding="utf-8")), self.mosaic_answer(cand_0001="teen"))

    def test_three_failed_or_invalid_answers_in_a_row_switch_offline(self):
        self.replies["photo"] = [{"counts": None, "error": "ConnectionError: reset"}, RuntimeError("boom"),
                                 {"counts": {"adults": 1}, "error": None}]
        state = self.state(modes=("photo",))
        errors = [self.ask(state, self.item(sha=f"{n:02d}" * 32))["errors"] for n in range(4)]
        self.assertEqual(errors, [["photo: ConnectionError: reset"], ["photo: RuntimeError: boom"],
                                  ["photo: invalid answer"], None])
        self.assertEqual((state["failures"], state["consecutive"], state["online"]), (3, 3, False))
        self.assertEqual(len(self.calls), 3)   # the fourth photo is not asked
        self.assertEqual(self.stderr.getvalue().count("Age model failed 3 times in a row"), 1)
        self.assertIn("Last error: photo: invalid answer", self.stderr.getvalue())
        self.assertEqual(self.cached_files(), [])

    def test_a_valid_answer_resets_the_streak(self):
        self.replies["photo"] = [RuntimeError("a"), RuntimeError("b"), self.photo_answer(), RuntimeError("c"),
                                 RuntimeError("d")]
        state = self.state(modes=("photo",))
        for n in range(5):
            self.ask(state, self.item(sha=f"{n:02d}" * 32))
        self.assertEqual((state["failures"], state["consecutive"], state["online"]), (4, 2, True))
        self.assertEqual(self.stderr.getvalue(), "")

    def test_gated_empty_frames_skip_photo_mode(self):
        answer = self.ask(self.state(modes=("photo",)), self.item(people=(), gated=True))
        self.assertEqual(answer, {"model": None, "photo": None, "persons": None, "errors": None})
        # With mosaic too there is still no request: no people means an empty (answered) mosaic.
        answer = self.ask(self.state(), self.item(people=(), gated=True))
        self.assertEqual(answer, {"model": "gemma4:26b", "photo": None, "persons": {"ages": {}, "error": None},
                                  "errors": None})
        self.assertEqual((self.calls, self.cached_files()), ([], []))
        # Not gated and without people: the photo is still counted.
        self.replies["photo"] = [self.photo_answer(0)]
        answer = self.ask(self.state(), self.item(people=()))
        self.assertEqual(self.calls, [("photo", (640, 480))])
        self.assertEqual(answer["photo"], self.photo_answer(0))

    def test_never_raises(self):
        state = self.state(modes=("photo",))
        broken = self.root / "broken.jpg"
        broken.write_bytes(b"not a jpeg")
        answer = self.ask(state, self.item(), path=broken)
        self.assertEqual(answer["model"], None)
        self.assertEqual(len(answer["errors"]), 1)
        self.assertTrue(answer["errors"][0].startswith("photo: UnidentifiedImageError"), answer["errors"])
        self.assertEqual(state["failures"], 1)
        # A cache that cannot be written keeps the answer and reports the write error.
        self.replies["photo"] = [self.photo_answer(2)]
        with patch("trailcam.storage.atomic_json", side_effect=OSError("disk full")):
            answer = self.ask(state, self.item())
        self.assertEqual((answer["model"], answer["photo"]), ("gemma4:26b", self.photo_answer(2)))
        self.assertEqual(answer["errors"], ["photo: cache write failed: disk full"])
        self.assertEqual((state["failures"], state["consecutive"]), (1, 0))
        self.assertEqual(self.cached_files(), [])

    def test_answer_validation(self):
        valid = self.photo_answer(2, 1)
        for answer, mode, ok in (
                (valid, "photo", True), ({**valid, "error": ""}, "photo", True),
                ({**valid, "error": "x"}, "photo", False), ({"counts": None}, "photo", False),
                ({"counts": {**valid["counts"], "teens": -1}}, "photo", False),
                ({"counts": {**valid["counts"], "teens": 1.0}}, "photo", False),
                ({"counts": {**valid["counts"], "teens": False}}, "photo", False),
                ({"counts": {k: v for k, v in valid["counts"].items() if k != "unclear"}}, "photo", False),
                (None, "photo", False), ([valid], "photo", False),
                ({"ages": {}}, "mosaic", True), ({"ages": {"p": "unclear", "q": "teen"}}, "mosaic", True),
                ({"ages": {"p": "Child"}}, "mosaic", False), ({"ages": {"p": None}}, "mosaic", False),
                ({"ages": None}, "mosaic", False), ({"ages": {"p": "adult"}, "error": "x"}, "mosaic", False),
                ({}, "mosaic", False), ("ages", "mosaic", False)):
            with self.subTest(answer=answer, mode=mode):
                self.assertIs(entry._valid_vlm_answer(answer, mode), ok)


if __name__ == "__main__":
    unittest.main()
