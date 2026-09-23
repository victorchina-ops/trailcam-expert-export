"""Synthetic checks for CSV rows (no models, no photos).

Covers the image / event / camera-day field sets, blank (unsupported or
unassessable) versus zero (assessed and absent) semantics, the compact
persons_json cell, the evidence confidence threshold, and spreadsheet safety
as applied by ``storage.write_csv``.
"""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timedelta
import json
import math
import tempfile
import unittest
from pathlib import Path

from trailcam import events as events_module
from trailcam import roster
from trailcam.export import (BASE_FIELDS, CELL_LIMIT, COUNT_CLASSES, EVENT_FIELDS, EVENT_NOTE, EVIDENCE_CONFIDENCE,
                             FIELDS, IMAGE_NOTE, MAIN_FIELDS, OBJECT_FIELDS, SUMMARY_FIELDS, count_class, json_cell,
                             make_event_row, make_row, summary_rows)
from trailcam.fusion import COUNT_FIELDS, DIRECTIONS, ORIENTATIONS, TRAVEL_DIRECTIONS, fuse
from trailcam.postprocess import postprocess
from trailcam.storage import spreadsheet_safe, write_csv

T0 = datetime(2024, 5, 1, 10, 0, 0)
COMPACT_KEYS = {"id", "box", "conf", "experts", "age", "age_method", "height_ratio", "height_px", "complete",
                "direction", "direction_source", "lateral", "radial", "facing", "orientation", "backpack",
                "large_bag", "track", "near", "near_source", "attributes", "pose", "associated_backpack",
                "backpack_source", "track_ambiguous", "track_review_reasons"}


def at(seconds):
    return (T0 + timedelta(seconds=seconds)).isoformat()


def det(label, confidence, box, **extra):
    return {"label": label, "confidence": confidence, "xyxy": list(box), "bbox_xyxy": list(box), **extra}


def standing_keypoints(box, confidence=0.9):
    x1, y1, x2, y2 = box
    w, h, cx = x2 - x1, y2 - y1, (x1 + x2) / 2
    layout = [(0, .08), (-.05, .06), (.05, .06), (-.1, .07), (.1, .07), (-.2, .2), (.2, .2),
              (-.25, .35), (.25, .35), (-.25, .5), (.25, .5), (-.15, .5), (.15, .5),
              (-.15, .75), (.15, .75), (-.15, .97), (.15, .97)]
    return [[cx + dx * w, y1 + dy * h, confidence] for dx, dy in layout]


def look(colour):
    upper, lower = [0.0] * 256, [0.0] * 256
    upper[colour], lower[colour + 1] = 1.0, 1.0
    return {"version": "hsv_v1", "upper": upper, "lower": lower, "valid": True}


def image(people=(), time=None, extra=None, width=2000, height=1200, strip_bottom=0):
    """Cached v2 inference result; ``people`` are (box, yoloe confidence) pairs; ``extra`` maps expert -> detections."""
    experts = {name: {"detections": [], "device": "cpu", "checkpoint": name + ".pt"}
               for name in ("yolo26n", "megadetector", "yoloe", "pose")}
    experts["pose"]["keypoint_names"] = ["nose"] * 17
    for box, confidence in people:
        experts["yoloe"]["detections"].append(det("person", confidence, box))
        experts["pose"]["detections"].append(det("person", .9, box, keypoints=standing_keypoints(box),
                                                 orientation_evidence={"orientation": "side",
                                                                       "facing_direction": "left"}))
    for name, dets in (extra or {}).items():
        experts[name]["detections"].extend(dets)
    for expert in experts.values():
        for index, d in enumerate(expert["detections"]):
            d["detection_index"] = index
    strip = {"top": 0, "bottom": strip_bottom}
    persons = []
    for c in roster.build_candidates(experts, strip, height):
        person = {"person_id": c["candidate_id"], **c, "keypoints": None, "pose_confidence": None,
                  "orientation_evidence": None, "mask_polygon_xy": None, "appearance": look(3),
                  "attributes": {"status": "ok", "native_orientation_label": "side", "backpack_presence": None}}
        if "pose" in c["members"]:
            source = experts["pose"]["detections"][c["members"]["pose"]["detection_index"]]
            person.update(keypoints=source["keypoints"], pose_confidence=source["confidence"],
                          orientation_evidence=source["orientation_evidence"])
        persons.append(person)
    return {"status": "ok", "error": None, "width": width, "height": height, "device": "cpu",
            "attribute_device": "cpu", "profile": "standard", "gated_empty": False, "analysis_seconds": 1.5,
            "analyzed_at_utc": "2024-05-01T08:00:00+00:00", "data_strip": strip,
            "metadata": {"capture_time": time, "time_source": "exif_original", "camera_make": "Acme",
                         "camera_model": "TC-1", "sequence_number": 7, "clock_suspect": False},
            "experts": experts, "persons": persons, "attribute_runtime": {"device": "cpu"},
            "timings": {"vision_total_seconds": 1.2}}


def metadata(image_id="img_1", relative_path="camA/IMG_0001.JPG"):
    return {"run_id": "run_1", "image_id": image_id, "relative_path": relative_path, "sha256": "ab" * 32,
            "cache_hit": False, "configuration_hash": "cfg", "current_run_seconds": 0.5}


def processed(*results, paths=None, settings=None):
    items = [{"relative_path": (paths or [f"camA/IMG_{n:04d}.JPG" for n in range(len(results))])[n],
              "image_id": f"img_{n}", "result": r} for n, r in enumerate(results)]
    post = postprocess(items, {"events": True, "geometry_age": True, "large_bags": True,
                               "near_fraction": .07, **(settings or {})})
    return items, post


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class FieldSetTests(unittest.TestCase):
    def test_no_duplicate_columns(self):
        for fields in (FIELDS, EVENT_FIELDS, SUMMARY_FIELDS):
            self.assertEqual(len(fields), len(set(fields)))

    def test_image_fields(self):
        self.assertEqual(FIELDS[:len(BASE_FIELDS)], BASE_FIELDS)
        self.assertTrue(set(MAIN_FIELDS) <= set(FIELDS))
        expected = {"event_id", "event_frame", "event_image_count", "camera_id", "camera_calibration_status",
                    "needs_review", "persons_json", "objects_json", "expert_evidence_json", "notes",
                    "direction_from_motion", "direction_from_facing", "large_bags", "large_bags_uncertain",
                    "candidates_total", "candidates_rejected", "age_by_camera_calibration", "age_by_relative_height"}
        expected |= {f"dir_{d}" for d in TRAVEL_DIRECTIONS} | {f"facing_{d}" for d in DIRECTIONS}
        expected |= {f"orientation_{o}" for o in ORIENTATIONS} | {f"unverified_{f}" for f in OBJECT_FIELDS}
        expected |= {f"{e}_{f}" for e in ("yoloe", "yolo26n", "megadetector") for f in COUNT_FIELDS}
        expected |= {f"{p}_{f}" for p in ("vote", "mean", "vote_status", "vote_model_count", "vote_spread")
                     for f in COUNT_FIELDS}
        self.assertTrue(expected <= set(FIELDS), expected - set(FIELDS))
        self.assertTrue(set(COUNT_FIELDS) <= set(FIELDS))

    def test_event_fields_cover_event_summaries(self):
        self.assertTrue(set(events_module.SUMMARY_FIELDS) <= set(EVENT_FIELDS),
                        set(events_module.SUMMARY_FIELDS) - set(EVENT_FIELDS))
        # people_near_* are added by postprocess (near zone), not by events.summarize_event.
        self.assertEqual(set(EVENT_FIELDS) - set(events_module.SUMMARY_FIELDS),
                         {"run_id", "notes", "people_near_max_frame", "people_near_unique",
                          "people_unique_class", "people_near_unique_class", "vlm_adults_max_frame",
                          "vlm_teens_max_frame", "vlm_children_max_frame", "vlm_unclear_max_frame"})

    def test_summary_fields_match_rows(self):
        rows = summary_rows([{"camera_id": "camA", "start": at(0), "image_count": 1}], "run_1")
        self.assertEqual(set(rows[0]), set(SUMMARY_FIELDS))


class MakeRowTests(unittest.TestCase):
    def test_processed_row_has_exact_fields(self):
        items, _ = processed(image([([500, 400, 600, 700], .9)], at(0)))
        row = make_row(items[0]["result"], metadata())
        self.assertEqual(list(row), FIELDS)
        self.assertEqual(row["run_id"], "run_1")
        self.assertEqual(row["notes"], IMAGE_NOTE)

    def test_image_note_explains_experimental_outputs_and_unknowns(self):
        self.assertIn("Adult/child counts preserve unknown by default", IMAGE_NOTE)
        self.assertIn("experimental opt-ins", IMAGE_NOTE)
        self.assertIn("presentation proxy is not gender identity", IMAGE_NOTE)
        self.assertIn("people_near equals all people unless", IMAGE_NOTE)
        self.assertIn("Blank means unsupported/disabled/unavailable", IMAGE_NOTE)
        self.assertLess(len(IMAGE_NOTE), 32767)   # one spreadsheet cell

    def test_error_row_is_blank_not_zero(self):
        row = make_row({"status": "error", "error": "OSError: bad file"}, metadata())
        self.assertEqual(list(row), FIELDS)
        self.assertEqual((row["status"], row["error"]), ("error", "OSError: bad file"))
        for field in MAIN_FIELDS + [f"facing_{d}" for d in DIRECTIONS] + ["persons_json", "objects_json",
                                                                           "expert_evidence_json", "event_id"]:
            self.assertIsNone(row[field], field)
        self.assertEqual(row["notes"], IMAGE_NOTE)
        self.assertEqual(row["image_id"], "img_1")

    def test_postprocess_failure_row_is_blank(self):
        result = image([([500, 400, 600, 700], .9)], at(0))
        fuse(result)
        result.update(status="error", error="Post-processing failed: fusion: KeyError: 'x'",
                      postprocess_error="fusion: KeyError: 'x'")
        row = make_row(result, metadata())
        self.assertIsNone(row["people_total"])
        self.assertIsNone(row["persons_json"])
        self.assertEqual(row["error"], "Post-processing failed: fusion: KeyError: 'x'")

    def test_ok_without_fusion_is_blank(self):
        row = make_row(image([([500, 400, 600, 700], .9)], at(0)), metadata())
        self.assertIsNone(row["people_total"])
        self.assertEqual(row["capture_time"], at(0))
        self.assertEqual(row["vision_device"], "cpu")

    def test_empty_image_counts_are_zero(self):
        items, _ = processed(image(time=at(0)))
        row = make_row(items[0]["result"], metadata())
        for field in MAIN_FIELDS:
            # Count-class columns are labels ("0", "1", "3–4", ...), not counts.
            self.assertEqual(row[field], "0" if field.endswith("_class") else 0, field)
        for field in OBJECT_FIELDS:
            self.assertEqual(row[f"unverified_{field}"], 0)
        self.assertEqual((row["candidates_total"], row["candidates_rejected"]), (0, 0))
        self.assertEqual(row["persons_json"], "[]")
        self.assertEqual(row["objects_json"], "{}")
        self.assertIs(row["needs_review"], False)

    def test_unsupported_expert_counts_are_blank(self):
        items, _ = processed(image(time=at(0)))
        row = make_row(items[0]["result"], metadata())
        self.assertEqual(row["yoloe_strollers"], 0)
        self.assertIsNone(row["yolo26n_strollers"])      # yolo26n has no stroller class
        self.assertIsNone(row["megadetector_bicycles"])  # MegaDetector: animal / person / vehicle only
        self.assertEqual(row["megadetector_people_total"], 0)
        self.assertIsNone(row["vote_strollers"])         # one eligible expert: no majority value
        self.assertEqual(row["vote_status_strollers"], "single_expert")
        self.assertEqual(row["vote_model_count_strollers"], 1)
        self.assertEqual(row["mean_strollers"], 0)
        self.assertEqual(row["vote_people_total"], 0)

    def test_fused_but_not_postprocessed_leaves_age_and_direction_blank(self):
        result = image([([500, 400, 600, 700], .9)], at(0))
        fuse(result)
        row = make_row(result, metadata())
        self.assertEqual(row["people_total"], 1)
        self.assertEqual(row["age_unknown"], 1)
        for field in ["adults", "children", "event_id", "camera_id", "direction_from_motion"] + \
                     [f"dir_{d}" for d in TRAVEL_DIRECTIONS]:
            self.assertIsNone(row[field], field)

    def test_postprocessed_values(self):
        items, post = processed(image([([500, 400, 600, 700], .9)], at(0), strip_bottom=30))
        row = make_row(items[0]["result"], metadata())
        self.assertEqual(row["camera_id"], "camA")
        self.assertEqual(row["event_id"], post["events"][0]["event_id"])
        self.assertEqual((row["event_frame"], row["event_image_count"]), (1, 1))
        self.assertEqual(row["camera_calibration_status"], "insufficient")
        self.assertEqual((row["people_total"], row["age_unknown"], row["adults"], row["children"]), (1, 1, 0, 0))
        self.assertEqual({f"dir_{d}": row[f"dir_{d}"] for d in TRAVEL_DIRECTIONS},
                         {f"dir_{d}": int(d == "left") for d in TRAVEL_DIRECTIONS})
        self.assertEqual((row["direction_from_facing"], row["direction_from_motion"]), (1, 0))
        self.assertEqual(row["facing_left"], 1)
        self.assertEqual(row["orientation_side"], 1)
        self.assertEqual(row["carrying_backpack_unknown"], 1)
        self.assertEqual((row["sequence_number"], row["time_source"], row["clock_suspect"]), (7, "exif_original", False))
        self.assertEqual(row["data_strip_bottom"], 30)
        self.assertEqual((row["profile"], row["gated_empty"], row["analysis_seconds"]), ("standard", False, 1.5))
        self.assertEqual(json.loads(row["timings_json"]), {"vision_total_seconds": 1.2})
        self.assertEqual(json.loads(row["attribute_runtime_json"]), {"device": "cpu"})

    def test_candidate_counts(self):
        items, _ = processed(image([([500, 400, 600, 700], .9)], at(0),
                                   extra={"yolo26n": [det("person", .3, [1500, 400, 1600, 700])]}))
        result = items[0]["result"]
        # v2: post-processing drops rejected candidates and keeps their numbers in candidate_counts.
        self.assertEqual(result["candidate_counts"], {"total": 2, "rejected": 1})
        self.assertEqual(len(result["persons"]), 1)
        row = make_row(result, metadata())
        self.assertEqual((row["candidates_total"], row["candidates_rejected"], row["people_total"]), (2, 1, 1))

    def test_candidate_counts_fall_back_to_the_persons_list(self):
        result = image([([500, 400, 600, 700], .9)], at(0), extra={"yolo26n": [det("person", .3, [1500, 400, 1600, 700])]})
        fuse(result)  # fused but not compacted: no candidate_counts yet
        self.assertNotIn("candidate_counts", result)
        row = make_row(result, metadata())
        self.assertEqual((row["candidates_total"], row["candidates_rejected"]), (2, 1))
        result["candidate_counts"] = {"total": 7, "rejected": 5}
        row = make_row(result, metadata())
        self.assertEqual((row["candidates_total"], row["candidates_rejected"]), (7, 5))


class PersonsJsonTests(unittest.TestCase):
    def setUp(self):
        extra = {"yolo26n": [det("person", .3, [1500, 400, 1600, 700])],   # rejected candidate
                 "megadetector": [det("person", .8, [500.4, 399.6, 600.2, 700.49])]}
        self.items, _ = processed(image([([500.4, 399.6, 600.2, 700.49], .87654)], at(0), extra=extra))
        self.row = make_row(self.items[0]["result"], metadata())
        self.people = json.loads(self.row["persons_json"])

    def test_only_accepted_people(self):
        self.assertEqual(len(self.people), 1)
        self.assertEqual(self.people[0]["id"], "cand_0001")

    def test_compact_keys_and_rounding(self):
        person = self.people[0]
        self.assertEqual(set(person), COMPACT_KEYS)
        self.assertEqual(person["box"], [500, 400, 600, 700])
        self.assertTrue(all(isinstance(v, int) for v in person["box"]))
        self.assertEqual(person["conf"], .9)                             # max over members
        self.assertEqual(person["experts"], {"yoloe": .88, "pose": .9})  # MegaDetector is not a people expert
        self.assertEqual(person["age"], "unknown")
        self.assertEqual((person["direction"], person["direction_source"]), ("left", "facing"))
        self.assertEqual(person["facing"], "left")
        self.assertTrue(person["track"].startswith("camA__"))
        self.assertIsInstance(person["height_px"], float)
        self.assertTrue(person["complete"])
        self.assertIsNone(person["lateral"])  # single observation: no motion

    def test_heavy_evidence_is_left_out(self):
        text = self.row["persons_json"]
        for key in ("keypoints", "appearance", "members", "mask_polygon_xy", "upper"):
            self.assertNotIn(f'"{key}"', text)
        self.assertLess(len(text), 2000)
        self.assertIn("attributes", self.people[0])
        self.assertIn("pose", self.people[0])

    def test_compact_separators(self):
        self.assertEqual(self.row["persons_json"], json.dumps(self.people, separators=(",", ":"), ensure_ascii=False))
        self.assertNotIn(", ", self.row["persons_json"])
        self.assertNotIn('": ', self.row["persons_json"])


class NativeExpertAndDisabledExportTests(unittest.TestCase):
    def test_native_attribute_labels_scores_and_pose_survive_a_default_csv_export(self):
        from trailcam.attributes import INDEX_NAMES, derive_attributes
        result = image([([100 + 300 * n, 400, 200 + 300 * n, 700], .9) for n in range(3)], at(0))
        expected = []
        for person, age, orientation, female in zip(result["persons"],
                ("age_under18", "age_18_60", "age_over60"), ("front", "side", "back"), (.95, .05, .5)):
            scores = dict.fromkeys(INDEX_NAMES.values(), .01)
            scores.update({age: .93, orientation: .91, "native_female": female, "backpack": .5})
            attributes = {"status": "ok", **derive_attributes(scores, 100, 300)}
            person["attributes"] = attributes
            expected.append(copy.deepcopy(attributes))
        items = [{"relative_path": "camA/IMG_0001.JPG", "image_id": "img_1", "result": result}]
        postprocess(items)  # Public defaults preserve combined unknown, not the native labels.
        row = make_row(result, metadata())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "native.csv"
            write_csv(path, [row], FIELDS)
            written, = read_csv(path)
        self.assertEqual((written["people_total"], written["adults"], written["children"], written["age_unknown"]),
                         ("3", "0", "0", "3"))
        self.assertEqual([written["attribute_age_" + age] for age in ("under18", "18_60", "over60", "unknown")],
                         ["1", "1", "1", "0"])
        self.assertEqual([written["attribute_orientation_" + orientation] for orientation in ORIENTATIONS],
                         ["1", "1", "1", "0"])
        self.assertEqual([written["attribute_presentation_proxy_" + label]
                          for label in ("feminine", "masculine", "unclear")], ["1", "1", "1"])
        self.assertEqual((written["pose_people_total"], written["pose_orientation_side"], written["pose_direction_left"]),
                         ("3", "3", "3"))
        for field in COUNT_FIELDS:
            self.assertTrue(written["combined_source_" + field])
        people = json.loads(written["persons_json"])
        self.assertEqual([p["attributes"] for p in people], expected)
        self.assertEqual([p["pose"] for p in people],
                         [{"orientation": "side", "facing_direction": "left", "confidence": .9}] * 3)
        self.assertTrue(all(p["age"] == "unknown" and p["age_method"] == "disabled" for p in people))

    def test_default_disabled_outputs_are_blank_and_known_absences_remain_zero(self):
        results = [image([([500, 400, 600, 700], .9)], at(0)), image(time=at(5))]
        items = [{"relative_path": f"camA/IMG_{n}.JPG", "image_id": f"img_{n}", "result": r}
                 for n, r in enumerate(results)]
        post = postprocess(items)
        self.assertEqual(post["events"], [])
        rows = [make_row(item["result"], metadata(image_id=item["image_id"])) for item in items]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "defaults.csv"
            write_csv(path, rows, FIELDS)
            written = read_csv(path)
        for row, count in zip(written, ("1", "0")):
            self.assertEqual((row["people_total"], row["people_near"], row["age_unknown"]), (count, count, count))
            self.assertEqual((row["adults"], row["children"], row["dogs"]), ("0", "0", "0"))
            for field in ("large_bags", "large_bags_uncertain", "event_id", "event_frame", "event_image_count",
                          "age_model", "vlm_adults", "vlm_children"):
                self.assertEqual(row[field], "", field)
            for field in ("events_enabled", "geometry_age_enabled", "large_bags_enabled"):
                self.assertEqual(row[field], "False")
            self.assertEqual(row["large_bags_status"], "disabled")
            self.assertEqual(row["camera_calibration_status"], "disabled")
        person, = json.loads(written[0]["persons_json"])
        self.assertIsNone(person["large_bag"])
        self.assertIsNone(person["track"])

    def test_event_and_day_large_bag_estimates_stay_blank_when_disabled(self):
        items, post = processed(image([([500, 400, 600, 700], .9)], at(0)),
                                settings={"large_bags": False, "geometry_age": False})
        event, = [make_event_row(e, "r") for e in post["events"]]
        day, = summary_rows(post["events"], "r")
        self.assertIsNone(event["large_bags"])
        self.assertIsNone(event["large_bags_uncertain"])
        self.assertIsNone(day["large_bags"])
        self.assertEqual(event["people_unique"], 1)
        self.assertEqual(event["age_unknown"], 1)

    def test_pose_aggregates_use_the_same_eligible_detections_as_pose_total(self):
        extra = {"pose": [det("person", .9, [700, 1170, 760, 1200],
                               keypoints=standing_keypoints([700, 1170, 760, 1200]),
                               orientation_evidence={"orientation": "front", "facing_direction": "toward"}),
                          det("person", .2, [1000, 400, 1100, 700],
                               keypoints=standing_keypoints([1000, 400, 1100, 700]),
                               orientation_evidence={"orientation": "back", "facing_direction": "away"})]}
        raw = image([([500, 400, 600, 700], .9)], at(0), extra=extra, strip_bottom=40)
        for compacted in (False, True):
            with self.subTest(compacted=compacted):
                result = copy.deepcopy(raw)
                if compacted:
                    items, _ = processed(result)
                else:
                    fuse(result)
                row = make_row(result, metadata())
                self.assertEqual(row["pose_people_total"], 1)
                self.assertEqual(sum(row["pose_orientation_" + label] for label in ORIENTATIONS), 1)
                self.assertEqual(sum(row["pose_direction_" + label] for label in DIRECTIONS), 1)
                self.assertEqual((row["pose_orientation_side"], row["pose_direction_left"]), (1, 1))

    def test_skipped_pose_outputs_are_blank_not_zero(self):
        result = image(time=at(0))
        result["experts"]["pose"]["skipped"] = "empty_frame_gate"
        items, _ = processed(result)
        row = make_row(result, metadata())
        fields = ["pose_people_total"] + ["pose_orientation_" + label for label in ORIENTATIONS]
        fields += ["pose_direction_" + label for label in DIRECTIONS]
        for field in fields:
            self.assertIsNone(row[field], field)

    def test_same_height_peers_do_not_turn_native_model_ages_into_combined_adults(self):
        result = image([([100, 400, 200, 700], .9), ([400, 400, 500, 700], .9)], at(0))
        for person in result["persons"]:
            person["attributes"]["native_age_label"] = "18_60"
        items, _ = processed(result)
        row = make_row(result, metadata())
        self.assertEqual(row["attribute_age_18_60"], 2)
        self.assertEqual((row["adults"], row["children"], row["age_unknown"]), (0, 0, 2))
        self.assertTrue(all(p["age"] == "unknown" for p in json.loads(row["persons_json"])))


class JsonCellTests(unittest.TestCase):
    def test_compact_unicode_and_strict(self):
        self.assertEqual(json_cell({"a": [1, 2], "b": "עין"}), '{"a":[1,2],"b":"עין"}')
        with self.assertRaises(ValueError):
            json_cell({"x": math.nan})
        with self.assertRaises(ValueError):
            json_cell([math.inf])


class ObjectsAndEvidenceTests(unittest.TestCase):
    def setUp(self):
        t = EVIDENCE_CONFIDENCE
        extra = {"yoloe": [det("bicycle", t, [100.04, 100.06, 200.44, 180.0]),
                           det("bicycle", t - 1e-4, [800, 100, 900, 180]),
                           det("dog", .12345, [1000, 100, 1100, 180])],
                 "megadetector": [det("animal", .9, [1300, 100, 1400, 180])]}
        self.items, _ = processed(image(time=at(0), extra=extra))
        self.row = make_row(self.items[0]["result"], metadata())

    def test_threshold_value(self):
        self.assertEqual(EVIDENCE_CONFIDENCE, 0.25)

    def test_evidence_keeps_detections_at_or_above_threshold(self):
        evidence = json.loads(self.row["expert_evidence_json"])
        self.assertEqual(set(evidence), {"yolo26n", "megadetector", "yoloe", "pose"})
        self.assertEqual(evidence["yoloe"]["detections"], [["bicycle", 0.25, 100.0, 100.1, 200.4, 180.0]])
        self.assertEqual(evidence["megadetector"]["detections"], [["animal", 0.9, 1300, 100, 1400, 180]])
        self.assertEqual(evidence["pose"]["detections"], [])

    def test_evidence_drops_bulky_keys_but_keeps_expert_metadata(self):
        evidence = json.loads(self.row["expert_evidence_json"])
        self.assertNotIn("keypoint_names", evidence["pose"])
        self.assertEqual(evidence["yoloe"]["checkpoint"], "yoloe.pt")
        self.assertEqual(evidence["yoloe"]["device"], "cpu")
        self.assertEqual(evidence["yoloe"]["counts"]["bicycles"], 1)  # counts at the counting threshold

    def test_objects_json_lists_only_fields_with_objects(self):
        objects = json.loads(self.row["objects_json"])
        self.assertEqual(set(objects), {"bicycles"})  # the MegaDetector animal alone creates no dog
        self.assertEqual(objects["bicycles"][0]["status"], "single_expert_confident")
        self.assertEqual((self.row["bicycles"], self.row["dogs"]), (1, 0))


class EventRowTests(unittest.TestCase):
    def test_event_row_from_postprocess(self):
        walk = [image([([100 + 200 * n, 500, 200 + 200 * n, 800], .9)], at(5 * n)) for n in range(3)]
        items, post = processed(*walk)
        row = make_event_row(post["events"][0], "run_1")
        self.assertEqual(list(row), EVENT_FIELDS)
        self.assertEqual(row["run_id"], "run_1")
        self.assertEqual(row["notes"], EVENT_NOTE)
        # v2: events identify photos by relative path, not by (content) image_id.
        paths = [item["relative_path"] for item in items]
        self.assertEqual(json.loads(row["images"]), paths)
        self.assertEqual(row["images"], '["camA/IMG_0000.JPG","camA/IMG_0001.JPG","camA/IMG_0002.JPG"]')
        self.assertEqual((row["image_count"], row["people_unique"], row["people_max_frame"]), (3, 1, 1))
        self.assertEqual((row["dir_right"], row["direction_from_motion"]), (1, 1))
        self.assertEqual(row["review_reasons"], "")
        self.assertIs(row["needs_review"], False)
        self.assertEqual((row["bicycles"], row["dogs"], row["large_bags"]), (0, 0, 0))

    def test_review_reasons_join_and_unknown_keys_dropped(self):
        event = {"event_id": "e1", "camera_id": "camA", "review_reasons": ["b_reason", "a_reason"],
                 "needs_review": True, "images": ["i1"], "tracks": [{"x": 1}], "people_unique": 2}
        row = make_event_row(event, "run_9")
        self.assertEqual(list(row), EVENT_FIELDS)
        self.assertEqual(row["review_reasons"], "b_reason; a_reason")
        self.assertNotIn("tracks", row)
        self.assertEqual(row["people_unique"], 2)
        self.assertEqual(make_event_row({**event, "review_reasons": "already text"}, "r")["review_reasons"],
                         "already text")
        self.assertIsNone(make_event_row({"event_id": "e2"}, "r")["review_reasons"])

    def test_unassessed_event_fields_stay_blank(self):
        event = events_module.summarize_event([{"image_id": "i1", "camera_id": "camA", "capture_time": at(0),
                                                "objects": {"dogs": None, "bicycles": 0}}])
        row = make_event_row(event, "r")
        self.assertIsNone(row["people_unique"])   # no frame carried a people list
        self.assertIsNone(row["dogs"])            # unsupported
        self.assertEqual(row["bicycles"], 0)      # assessed, absent
        self.assertEqual(row["images"], '["i1"]')

    def test_input_not_mutated(self):
        event = {"event_id": "e1", "review_reasons": ["x"], "images": ["i1"]}
        before = copy.deepcopy(event)
        make_event_row(event, "r")
        self.assertEqual(event, before)

    def test_images_cell_is_truncated_above_the_spreadsheet_limit(self):
        self.assertEqual(CELL_LIMIT, 32000)
        paths = [f"camA/day_{n // 1000:02d}/IMG_{n:05d}.JPG" for n in range(1500)]
        full = json_cell(paths)
        self.assertGreater(len(full), CELL_LIMIT)
        event = {"event_id": "camA__long", "images": paths, "image_count": len(paths)}
        before = copy.deepcopy(event)
        row = make_event_row(event, "r")
        self.assertLessEqual(len(row["images"]), CELL_LIMIT)
        self.assertEqual(json.loads(row["images"]),
                         {"first": paths[0], "last": paths[-1], "count": 1500, "truncated": True})
        self.assertEqual(event, before)
        # At or below the limit the full list is kept.
        fitting = paths[:1000]
        self.assertLessEqual(len(json_cell(fitting)), CELL_LIMIT)
        self.assertEqual(json.loads(make_event_row({**event, "images": fitting}, "r")["images"]), fitting)
        boundary = ["x" * (CELL_LIMIT - 4)]  # '["' + text + '"]' is exactly CELL_LIMIT characters
        self.assertEqual(len(json_cell(boundary)), CELL_LIMIT)
        self.assertEqual(json.loads(make_event_row({"images": boundary}, "r")["images"]), boundary)
        longer = ["x" * (CELL_LIMIT - 3)]
        self.assertEqual(json.loads(make_event_row({"images": longer}, "r")["images"])["count"], 1)


class SummaryRowTests(unittest.TestCase):
    def events(self):
        base = {"image_count": 2, "people_unique": 3, "people_max_frame": 2, "adults": 2, "children": 1,
                "age_unknown": 0, "large_bags": 1, "bicycles": 1, "strollers": 0, "motorcycles": 0, "atv_utv": 0,
                "other_vehicles": 0, "dogs": 1, **{f"dir_{d}": 0 for d in TRAVEL_DIRECTIONS}, "dir_left": 3}
        return [{**base, "camera_id": "camB", "start": "2024-05-01T10:00:00", "needs_review": True},
                {**base, "camera_id": "camA", "start": "2024-05-02T09:00:00", "needs_review": False},
                {**base, "camera_id": "camA", "start": "2024-05-01T23:59:59", "needs_review": True,
                 "dogs": None, "people_unique": 1, "people_max_frame": 1},
                {**base, "camera_id": "camA", "start": "2024-05-01T08:00:00+03:00", "needs_review": False},
                {**base, "camera_id": "camA", "start": None, "needs_review": False}]

    def test_field_order_puts_people_max_frame_before_people_unique(self):
        # v2: the camera-days CSV carries the conservative MaxN total next to people_unique.
        self.assertEqual(SUMMARY_FIELDS[:11], ["run_id", "camera_id", "date", "events", "images", "people_max_frame",
                                               "people_unique", "people_near_max_frame", "people_near_unique",
                                               "adults", "children"])

    def test_people_max_frame_is_summed_per_day(self):
        rows = {(r["camera_id"], r["date"]): r for r in summary_rows(self.events(), "run_1")}
        self.assertEqual(rows[("camA", "2024-05-01")]["people_max_frame"], 2 + 1)
        self.assertEqual(rows[("camB", "2024-05-01")]["people_max_frame"], 2)
        unassessed = summary_rows([{"camera_id": "camC", "start": at(0), "image_count": 1,
                                    "people_max_frame": None, "people_unique": None}], "r")[0]
        self.assertEqual((unassessed["people_max_frame"], unassessed["people_unique"]), (None, None))

    def test_grouped_by_camera_and_date_and_sorted(self):
        rows = summary_rows(self.events(), "run_1")
        self.assertEqual([(r["camera_id"], r["date"]) for r in rows],
                         [("camA", "2024-05-01"), ("camA", "2024-05-02"), ("camA", "unknown"), ("camB", "2024-05-01")])
        for row in rows:
            self.assertEqual(set(row), set(SUMMARY_FIELDS))
            self.assertEqual(row["run_id"], "run_1")

    def test_sums_deduplicated_event_totals(self):
        first = summary_rows(self.events(), "run_1")[0]
        self.assertEqual(first["events"], 2)
        self.assertEqual(first["images"], 4)
        self.assertEqual(first["people_unique"], 4)
        self.assertEqual((first["adults"], first["children"], first["dir_left"]), (4, 2, 6))
        self.assertEqual(first["dogs"], 1)  # the unassessed event adds nothing
        self.assertEqual(first["large_bags"], 2)
        self.assertEqual(first["events_needing_review"], 1)

    def test_empty(self):
        self.assertEqual(summary_rows([], "run_1"), [])


class SpreadsheetSafetyTests(unittest.TestCase):
    def test_spreadsheet_safe(self):
        # Leading whitespace does not hide a formula.
        for text in ("=1+1", "+1", "-1", "@SUM(A1)", "  =cmd", "\t=cmd", "\r\n=cmd", "\n@x"):
            self.assertEqual(spreadsheet_safe(text), "'" + text)
        # v2: a leading tab / CR / LF is escaped by itself (a lstrip() used to hide it).
        for text in ("\tIMG_0001.JPG", "\rplain", "\nplain", "\t"):
            self.assertEqual(spreadsheet_safe(text), "'" + text)
        for value in ("camA/IMG.JPG", "[1,2]", '{"a":1}', -1, 0, None, False, 2.5, " plain", "a\tb", "x\n"):
            self.assertEqual(spreadsheet_safe(value), value)

    def test_write_csv_escapes_formulas_in_image_event_and_summary_rows(self):
        items, post = processed(image([([500, 400, 600, 700], .9)], at(0)),
                                paths=["-cam/=HYPERLINK(\"x\").jpg"])
        rows = [make_row(items[0]["result"], metadata(relative_path=items[0]["relative_path"]))]
        rows.append(make_row({"status": "error", "error": "=2+2"}, metadata(image_id="img_2",
                                                                              relative_path="@evil.jpg")))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "images.csv"
            write_csv(path, rows, FIELDS)
            self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))  # UTF-8 BOM for spreadsheets
            written = read_csv(path)
            self.assertEqual(list(written[0]), FIELDS)
            self.assertEqual(written[0]["relative_path"], "'-cam/=HYPERLINK(\"x\").jpg")
            self.assertEqual(written[0]["camera_id"], "'-cam")
            self.assertEqual(written[1]["relative_path"], "'@evil.jpg")
            self.assertEqual(written[1]["error"], "'=2+2")
            self.assertEqual(written[0]["people_total"], "1")      # zero / numbers untouched
            self.assertEqual(written[0]["children"], "0")
            self.assertEqual(written[1]["people_total"], "")      # blank stays blank
            self.assertEqual(json.loads(written[0]["persons_json"])[0]["id"], "cand_0001")
            self.assertTrue(written[0]["event_id"].startswith("'-cam__2024-05-01T10-00-00"), written[0]["event_id"])

            events_path = Path(tmp) / "events.csv"
            write_csv(events_path, [make_event_row(e, "run_1") for e in post["events"]], EVENT_FIELDS)
            event = read_csv(events_path)[0]
            self.assertEqual(event["camera_id"], "'-cam")
            self.assertEqual(event["people_unique"], "1")

            summary_path = Path(tmp) / "days.csv"
            write_csv(summary_path, summary_rows(post["events"], "run_1"), SUMMARY_FIELDS)
            day = read_csv(summary_path)[0]
            self.assertEqual((day["camera_id"], day["date"], day["people_unique"]), ("'-cam", "2024-05-01", "1"))

    def test_unknown_columns_are_refused_and_no_file_is_left(self):
        row = make_row({"status": "error", "error": "x"}, {**metadata(), "surprise": 1})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.csv"
            with self.assertRaises(ValueError):
                write_csv(path, [row], FIELDS)
            self.assertEqual(list(Path(tmp).iterdir()), [])


class EndToEndTests(unittest.TestCase):
    def test_images_csv_blank_versus_zero(self):
        ok, empty = image([([500, 400, 600, 700], .9)], at(0)), image(time=at(10))
        failed = {"status": "error", "error": "OSError: truncated"}
        items, post = processed(ok, empty, failed)
        rows = [make_row(it["result"], metadata(image_id=it["image_id"], relative_path=it["relative_path"]))
                for it in items]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "images.csv"
            write_csv(path, rows, FIELDS)
            written = read_csv(path)
        self.assertEqual([r["people_total"] for r in written], ["1", "0", ""])
        self.assertEqual([r["dogs"] for r in written], ["0", "0", ""])
        self.assertEqual([r["yolo26n_strollers"] for r in written], ["", "", ""])
        self.assertEqual([r["event_image_count"] for r in written], ["2", "2", ""])
        self.assertEqual(written[2]["status"], "error")
        self.assertEqual(len(post["events"]), 1)


class CountClassTests(unittest.TestCase):
    """Group-size classes 0, 1, 2, 3-4, 5-10, >10 (blank for unknown)."""

    def test_count_classes(self):
        expected = {0: "0", 1: "1", 2: "2", 3: "3–4", 4: "3–4", 5: "5–10", 10: "5–10", 11: ">10", 250: ">10"}
        for value, label in expected.items():
            self.assertEqual(count_class(value), label, value)
        self.assertIsNone(count_class(None))

    def test_every_count_up_to_twenty(self):
        labels = [count_class(n) for n in range(21)]
        self.assertEqual(labels, ["0", "1", "2", "3–4", "3–4"] + ["5–10"] * 6 + [">10"] * 10)

    def test_classes_are_contiguous_and_open_ended(self):
        self.assertEqual([label for _, _, label in COUNT_CLASSES], ["0", "1", "2", "3–4", "5–10", ">10"])
        self.assertEqual(COUNT_CLASSES[0][0], 0)
        for (_, high, _), (low, _, _) in zip(COUNT_CLASSES, COUNT_CLASSES[1:]):
            self.assertEqual(low, high + 1)
        self.assertIsNone(COUNT_CLASSES[-1][1])

    def test_negative_counts_have_no_class(self):
        self.assertIsNone(count_class(-1))

    def test_image_row_classes(self):
        for count, label in ((0, "0"), (1, "1"), (2, "2"), (4, "3–4"), (5, "5–10"), (11, ">10")):
            with self.subTest(count=count):
                people = [([60 + 150 * n, 400, 160 + 150 * n, 700], .9) for n in range(count)]
                items, _ = processed(image(people, at(0)))
                row = make_row(items[0]["result"], metadata())
                self.assertEqual((row["people_total"], row["people_class"]), (count, label))
                self.assertEqual((row["people_near"], row["people_near_class"]), (count, label))  # all near

    def test_event_row_classes(self):
        for unique, near, labels in ((0, 0, ("0", "0")), (3, 2, ("3–4", "2")), (10, 11, ("5–10", ">10")),
                                     (None, None, (None, None))):
            with self.subTest(unique=unique, near=near):
                row = make_event_row({"event_id": "e", "people_unique": unique, "people_near_unique": near}, "r")
                self.assertEqual((row["people_unique_class"], row["people_near_unique_class"]), labels)
        row = make_event_row({"event_id": "e"}, "r")
        self.assertEqual((row["people_unique_class"], row["people_near_unique_class"]), (None, None))

    def test_class_labels_survive_the_csv_writer(self):
        rows = [make_event_row({"event_id": f"e{n}", "people_unique": n, "people_near_unique": n}, "r")
                for n in (0, 3, 7, 12)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            write_csv(path, rows, EVENT_FIELDS)
            written = read_csv(path)
        self.assertEqual([r["people_unique_class"] for r in written], ["0", "3–4", "5–10", ">10"])
        self.assertEqual([r["people_near_unique_class"] for r in written], ["0", "3–4", "5–10", ">10"])


class NearAndVlmColumnTests(unittest.TestCase):
    COUNTS = {"people_total": 3, "adults": 1, "teens": 1, "children": 1, "unclear": 0}

    def test_image_columns_exist(self):
        for field in ("people_near", "people_class", "people_near_class"):
            self.assertIn(field, MAIN_FIELDS)
        for field in ("age_model", "vlm_adults", "vlm_teens", "vlm_children", "vlm_age_unclear", "age_by_vlm"):
            self.assertIn(field, FIELDS)
        self.assertEqual(SUMMARY_FIELDS[7:9], ["people_near_max_frame", "people_near_unique"])
        self.assertFalse({f for f in SUMMARY_FIELDS if f.startswith("vlm_") or f.endswith("_class")})

    def test_people_near_in_the_image_row(self):
        # A 300 px person and a 30 px person far up the frame (limit 0.07 * 1200 = 84 px).
        items, _ = processed(image([([500, 400, 600, 700], .9), ([1500, 100, 1510, 130], .9)], at(0)))
        row = make_row(items[0]["result"], metadata())
        self.assertEqual((row["people_total"], row["people_near"]), (2, 1))
        self.assertEqual((row["people_class"], row["people_near_class"]), ("2", "1"))
        people = sorted(json.loads(row["persons_json"]), key=lambda p: p["box"][0])
        self.assertEqual([p["near"] for p in people], [True, False])
        self.assertEqual({p["near_source"] for p in people}, {"height_at_similar_depth"})

    def test_vlm_columns_are_blank_without_an_age_model(self):
        items, _ = processed(image([([500, 400, 600, 700], .9)], at(0)))
        row = make_row(items[0]["result"], metadata())
        for field in ("age_model", "vlm_adults", "vlm_teens", "vlm_children", "vlm_age_unclear"):
            self.assertIsNone(row[field], field)

    def test_vlm_photo_counts_in_the_image_row(self):
        result = image([([500, 400, 600, 700], .9)], at(0))
        result["vlm_age"] = {"model": "gemma4:26b", "photo": {"counts": dict(self.COUNTS), "error": None},
                             "persons": {"ages": {"cand_0001": "child"}, "error": None}}
        items, _ = processed(result)
        row = make_row(items[0]["result"], metadata())
        self.assertEqual((row["age_model"], row["vlm_adults"], row["vlm_teens"], row["vlm_children"],
                          row["vlm_age_unclear"], row["age_by_vlm"]), ("gemma4:26b", 1, 1, 1, 0, 1))
        self.assertEqual((row["children"], row["adults"]), (1, 0))
        self.assertEqual(json.loads(row["persons_json"])[0]["age_method"], "vlm")

    def test_failed_photo_request_leaves_counts_blank(self):
        result = image([([500, 400, 600, 700], .9)], at(0))
        result["vlm_age"] = {"model": "gemma4:26b", "photo": {"counts": None, "error": "ConnectionError: x"},
                             "persons": None}
        items, _ = processed(result)
        row = make_row(items[0]["result"], metadata())
        self.assertEqual(row["age_model"], "gemma4:26b")
        for field in ("vlm_adults", "vlm_teens", "vlm_children", "vlm_age_unclear"):
            self.assertIsNone(row[field], field)

    def test_error_rows_leave_near_and_vlm_columns_blank(self):
        row = make_row({"status": "error", "error": "x", "vlm_age": {"model": "m"}}, metadata())
        for field in ("people_near", "people_class", "people_near_class", "age_model", "vlm_adults", "age_by_vlm"):
            self.assertIsNone(row[field], field)

    def test_event_row_passes_near_and_vlm_maxima_through(self):
        event = {"event_id": "e", "people_unique": 4, "people_near_max_frame": 2, "people_near_unique": 3,
                 "vlm_adults_max_frame": 2, "vlm_teens_max_frame": 0, "vlm_children_max_frame": 1,
                 "vlm_unclear_max_frame": None}
        row = make_event_row(event, "r")
        self.assertEqual({k: row[k] for k in event if k != "event_id"},
                         {k: v for k, v in event.items() if k != "event_id"})
        self.assertEqual((row["people_unique_class"], row["people_near_unique_class"]), ("3–4", "3–4"))

    def test_camera_days_sum_near_columns(self):
        events = [{"camera_id": "camA", "start": at(0), "image_count": 2, "people_near_max_frame": 2,
                   "people_near_unique": 3},
                  {"camera_id": "camA", "start": at(600), "image_count": 1, "people_near_max_frame": 1,
                   "people_near_unique": 1},
                  {"camera_id": "camA", "start": at(900), "image_count": 1, "people_near_max_frame": None,
                   "people_near_unique": None}]
        day, = summary_rows(events, "r")
        self.assertEqual((day["people_near_max_frame"], day["people_near_unique"], day["events"]), (3, 4, 3))

    def test_near_columns_from_postprocess_in_all_three_csvs(self):
        frames = [image([([100 + 200 * n, 500, 200 + 200 * n, 800], .9), ([1800, 100, 1810, 130], .9)], at(5 * n))
                  for n in range(2)]
        items, post = processed(*frames)
        with tempfile.TemporaryDirectory() as tmp:
            images_path, events_path, days_path = (Path(tmp) / name for name in ("i.csv", "e.csv", "d.csv"))
            write_csv(images_path, [make_row(it["result"], metadata(image_id=it["image_id"],
                                                                     relative_path=it["relative_path"]))
                                    for it in items], FIELDS)
            write_csv(events_path, [make_event_row(e, "r") for e in post["events"]], EVENT_FIELDS)
            write_csv(days_path, summary_rows(post["events"], "r"), SUMMARY_FIELDS)
            rows, event_rows, day_rows = read_csv(images_path), read_csv(events_path), read_csv(days_path)
        self.assertEqual([(r["people_total"], r["people_near"], r["people_class"], r["people_near_class"])
                          for r in rows], [("2", "1", "2", "1")] * 2)
        event, = event_rows
        self.assertEqual((event["people_unique"], event["people_near_max_frame"], event["people_near_unique"],
                          event["people_unique_class"], event["people_near_unique_class"]), ("2", "1", "1", "2", "1"))
        self.assertEqual(event["vlm_children_max_frame"], "")
        day, = day_rows
        self.assertEqual((day["people_near_max_frame"], day["people_near_unique"]), ("1", "1"))


if __name__ == "__main__":
    unittest.main()
