"""Synthetic multi-image checks for cross-image post-processing (no models, no photos).

Covers camera ids, calibration grouping (camera | make model @ size), age
classification counts (camera calibration and relative height), event
grouping with the 60 s default gap, cross-frame tracks and motion direction,
the facing fallback for single observations, the per-image ``post`` fields,
and isolation of a record whose fusion fails.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta
import json
import unittest
from unittest.mock import patch

from trailcam import fusion, roster
from trailcam.age_geometry import POLICY as AGE_POLICY
from trailcam.fusion import TRAVEL_DIRECTIONS
from trailcam.postprocess import (DEFAULTS, KEEP_CONFIDENCE, _camera_key,
                                 postprocess as public_postprocess, postprocess_copy as public_postprocess_copy,
                                 prepare as public_prepare)


# These tests explicitly exercise optional features; production defaults stay conservative.
EXPERIMENTAL_SETTINGS = {"events": True, "geometry_age": True, "large_bags": True, "near_fraction": .07}


def experimental_postprocess(items, settings=None):
    return public_postprocess(items, {**EXPERIMENTAL_SETTINGS, **(settings or {})})


def experimental_postprocess_copy(items, settings=None):
    return public_postprocess_copy(items, {**EXPERIMENTAL_SETTINGS, **(settings or {})})


def experimental_prepare(frame, settings=None):
    return public_prepare(frame, {**EXPERIMENTAL_SETTINGS, **(settings or {})})

WIDTH, HEIGHT = 2000, 1200
T0 = datetime(2024, 5, 1, 10, 0, 0)


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
    """One-hot appearance: identical colours -> distance 0, different colours -> 1."""
    upper, lower = [0.0] * 256, [0.0] * 256
    upper[colour], lower[colour + 1] = 1.0, 1.0
    return {"version": "hsv_v1", "upper": upper, "lower": lower, "valid": True}


def walker(box, colour=0, orientation="unknown", pose_orientation="unknown", facing="unclear", pose=True):
    return {"box": list(box), "colour": colour, "orientation": orientation, "pose_orientation": pose_orientation,
            "facing": facing, "pose": pose}


def image(people=(), time=None, seq=None, make="Acme", model="TC-1", width=WIDTH, height=HEIGHT, status="ok"):
    """One cached v2 inference result (YOLOE + pose per person) with capture metadata."""
    yoloe, pose = [], []
    for p in people:
        yoloe.append(det("person", .9, p["box"], detection_index=len(yoloe)))
        if p["pose"]:
            pose.append(det("person", .9, p["box"], detection_index=len(pose), keypoints=standing_keypoints(p["box"]),
                            orientation_evidence={"orientation": p["pose_orientation"],
                                                  "facing_direction": p["facing"]}))
    experts = {"yolo26n": {"detections": []}, "megadetector": {"detections": []},
               "yoloe": {"detections": yoloe}, "pose": {"detections": pose}}
    strip = {"top": 0, "bottom": 0}
    persons = []
    for c in roster.build_candidates(experts, strip, height):
        spec = next(p for p in people if p["box"] == c["xyxy"])
        person = {"person_id": c["candidate_id"], **c, "keypoints": None, "pose_confidence": None,
                  "orientation_evidence": None, "mask_polygon_xy": None, "appearance": look(spec["colour"]),
                  "attributes": {"status": "ok", "native_orientation_label": spec["orientation"],
                                 "backpack_presence": None}}
        if "pose" in c["members"]:
            source = pose[c["members"]["pose"]["detection_index"]]
            person.update(keypoints=source["keypoints"], pose_confidence=source["confidence"],
                          orientation_evidence=source["orientation_evidence"])
        persons.append(person)
    return {"status": status, "width": width, "height": height, "data_strip": strip,
            "metadata": {"capture_time": time, "time_source": "exif_original" if time else "file_mtime",
                         "camera_make": make, "camera_model": model, "sequence_number": seq,
                         "clock_suspect": False},
            "experts": experts, "persons": persons}


def item(relative_path, result, image_id=None):
    return {"relative_path": relative_path, "image_id": image_id or "img_" + relative_path.replace("/", "_"),
            "result": result}


def accepted(result):
    return [p for p in result["persons"] if p.get("accepted")]


class CameraKeyTests(unittest.TestCase):
    def test_camera_make_model_and_size(self):
        base = {"camera_id": "camA", "result": image()}
        self.assertEqual(_camera_key(base), "camA|Acme TC-1@2000x1200")
        self.assertEqual(_camera_key({**base, "result": image(make=None, model=None)}), "camA|unknown@2000x1200")
        self.assertEqual(_camera_key({**base, "result": image(make=" Acme ", model=None)}), "camA|Acme@2000x1200")
        self.assertEqual(_camera_key({**base, "result": image(width=1920, height=1080)}), "camA|Acme TC-1@1920x1080")
        no_metadata = image()
        no_metadata["metadata"] = None
        self.assertEqual(_camera_key({**base, "result": no_metadata}), "camA|unknown@2000x1200")


class WalkAndFacingTests(unittest.TestCase):
    """camA: one person walks right across 3 frames; camB: one frame, facing only."""

    @classmethod
    def setUpClass(cls):
        walk = [walker([100 + 200 * n, 500, 200 + 200 * n, 800], colour=10, orientation="front") for n in range(3)]
        cls.cam_a = [item(f"camA/IMG_000{n + 1}.JPG", image([walk[n]], at(5 * n), n + 1)) for n in range(3)]
        cls.cam_b = item("camB/IMG_0100.JPG", image([
            walker([300, 400, 400, 700], colour=20, orientation="side", pose_orientation="side", facing="left"),
            walker([1200, 400, 1300, 700], colour=40)], at(3600), 100))
        # Input order is deliberately not capture order.
        cls.items = [cls.cam_a[2], cls.cam_b, cls.cam_a[0], cls.cam_a[1]]
        cls.post = experimental_postprocess(cls.items)
        cls.events = {e["camera_id"]: e for e in cls.post["events"]}

    def test_return_shape_and_settings(self):
        self.assertEqual(set(self.post), {"events", "calibrations", "settings"})
        self.assertEqual(self.post["settings"], {**DEFAULTS, **EXPERIMENTAL_SETTINGS})
        self.assertEqual(len(self.post["events"]), 2)
        json.dumps(self.post, allow_nan=False)

    def test_camera_ids_from_folders(self):
        for it in self.cam_a:
            self.assertEqual(it["camera_id"], "camA")
            self.assertEqual(it["result"]["post"]["camera_id"], "camA")
        self.assertEqual(self.cam_b["result"]["post"]["camera_id"], "camB")

    def test_post_event_fields_follow_capture_order(self):
        event_ids = {it["result"]["post"]["event_id"] for it in self.cam_a}
        self.assertEqual(event_ids, {self.events["camA"]["event_id"]})
        self.assertTrue(self.events["camA"]["event_id"].startswith("camA__2024-05-01T10-00-00"))
        self.assertEqual([it["result"]["post"]["event_frame"] for it in self.cam_a], [1, 2, 3])
        self.assertEqual({it["result"]["post"]["event_image_count"] for it in self.cam_a}, {3})
        post_b = self.cam_b["result"]["post"]
        self.assertEqual((post_b["event_frame"], post_b["event_image_count"]), (1, 1))
        self.assertNotEqual(post_b["event_id"], self.events["camA"]["event_id"])
        # v2: events list their photos by relative path (byte-identical copies share an image_id).
        self.assertEqual(self.events["camA"]["images"], [it["relative_path"] for it in self.cam_a])

    def test_person_tracked_across_three_frames(self):
        people = [accepted(it["result"]) for it in self.cam_a]
        self.assertEqual([len(p) for p in people], [1, 1, 1])
        tracks = {p[0]["track_id"] for p in people}
        self.assertEqual(len(tracks), 1)
        self.assertTrue(next(iter(tracks)).startswith(self.events["camA"]["event_id"]))

    def test_direction_from_motion_overrides_facing(self):
        for it in self.cam_a:
            person = accepted(it["result"])[0]
            self.assertEqual(person["combined"]["facing"], "toward")          # attribute says front
            self.assertEqual(person["combined"]["direction"], "right")        # but the track moves right
            self.assertEqual(person["combined"]["direction_source"], "motion")
            self.assertEqual(person["motion"]["source"], "motion")
            self.assertGreater(person["motion"]["lateral"], 0.25)
            self.assertAlmostEqual(person["motion"]["seconds"], 10.0)
            combined = it["result"]["combined"]
            self.assertEqual(combined["direction_counts"], {d: int(d == "right") for d in TRAVEL_DIRECTIONS})
            self.assertEqual(list(combined["direction_counts"]), list(TRAVEL_DIRECTIONS))
            self.assertEqual((combined["direction_from_motion"], combined["direction_from_facing"]), (1, 0))

    def test_event_summary_deduplicates(self):
        event = self.events["camA"]
        self.assertEqual(sum(it["result"]["combined"]["counts"]["people_total"] for it in self.cam_a), 3)
        self.assertEqual((event["image_count"], event["people_unique"], event["people_max_frame"]), (3, 1, 1))
        self.assertEqual((event["dir_right"], event["direction_from_motion"], event["direction_from_facing"]), (1, 1, 0))
        self.assertEqual(event["duration_seconds"], 10.0)
        self.assertFalse(event["needs_review"])

    def test_single_observation_uses_facing(self):
        people = sorted(accepted(self.cam_b["result"]), key=lambda p: p["xyxy"][0])
        self.assertEqual([(p["combined"]["direction"], p["combined"]["direction_source"]) for p in people],
                         [("left", "facing"), ("unclear", "none")])
        combined = self.cam_b["result"]["combined"]
        self.assertEqual((combined["direction_from_motion"], combined["direction_from_facing"]), (0, 1))
        self.assertEqual(combined["direction_counts"]["left"], 1)
        self.assertEqual(combined["direction_counts"]["unclear"], 1)
        event = self.events["camB"]
        self.assertEqual((event["people_unique"], event["dir_left"], event["dir_unclear"]), (2, 1, 1))
        self.assertEqual(event["direction_from_facing"], 1)

    def test_calibration_per_camera_is_insufficient_with_few_people(self):
        self.assertEqual(set(self.post["calibrations"]), {"camA|Acme TC-1@2000x1200", "camB|Acme TC-1@2000x1200"})
        for calibration in self.post["calibrations"].values():
            self.assertEqual(calibration["status"], "insufficient")
        self.assertEqual(self.cam_a[0]["result"]["post"]["camera_calibration_status"], "insufficient")

    def test_lone_walker_has_no_age_evidence(self):
        combined = self.cam_a[0]["result"]["combined"]
        self.assertEqual((combined["adults"], combined["children"], combined["age_unknown"]), (0, 0, 1))
        self.assertEqual(combined["age_status"], "geometry_estimate")
        person = accepted(self.cam_a[0]["result"])[0]
        self.assertTrue(person["geometry"]["complete"], person["geometry"]["reasons"])
        self.assertEqual(person["age_evidence"]["method"], "none")


class StationaryAndLeftTests(unittest.TestCase):
    def test_stationary_and_left_walkers(self):
        still = [item(f"cam/{n}.jpg", image([walker([500, 400, 600, 700], colour=5)], at(5 * n), n)) for n in range(2)]
        left = [item(f"other/{n}.jpg", image([walker([900 - 250 * n, 400, 1000 - 250 * n, 700], colour=7)],
                                             at(3 * n), n)) for n in range(2)]
        experimental_postprocess(still + left)
        self.assertEqual(accepted(still[1]["result"])[0]["combined"]["direction"], "stationary")
        self.assertEqual(accepted(left[0]["result"])[0]["combined"]["direction"], "left")
        self.assertEqual(accepted(left[0]["result"])[0]["combined"]["direction_source"], "motion")

    def test_different_people_are_not_linked(self):
        frames = [item(f"cam/{n}.jpg", image([walker([500 + 20 * n, 400, 600 + 20 * n, 700], colour=5 + 50 * n)],
                                             at(2 * n), n)) for n in range(2)]
        post = experimental_postprocess(frames)
        self.assertEqual(post["events"][0]["people_unique"], 2)
        ids = {accepted(f["result"])[0]["track_id"] for f in frames}
        self.assertEqual(len(ids), 2)


class EventGapTests(unittest.TestCase):
    def frames(self, seconds):
        return [item(f"camA/IMG_{n:04d}.JPG", image(time=at(s), seq=n)) for n, s in enumerate(seconds)]

    def test_default_gap_is_sixty_seconds_inclusive(self):
        self.assertEqual(DEFAULTS["event_gap_seconds"], 60)
        frames = self.frames([0, 60, 121])
        post = experimental_postprocess(frames)
        self.assertEqual([e["image_count"] for e in post["events"]], [2, 1])
        self.assertEqual(frames[0]["result"]["post"]["event_id"], frames[1]["result"]["post"]["event_id"])
        self.assertNotEqual(frames[1]["result"]["post"]["event_id"], frames[2]["result"]["post"]["event_id"])
        self.assertEqual([f["result"]["post"]["event_frame"] for f in frames], [1, 2, 1])
        self.assertEqual([f["result"]["post"]["event_image_count"] for f in frames], [2, 2, 1])

    def test_gap_setting_overrides_default(self):
        post = experimental_postprocess(self.frames([0, 60, 121]), {"event_gap_seconds": 30})
        self.assertEqual([e["image_count"] for e in post["events"]], [1, 1, 1])
        self.assertEqual(post["settings"]["event_gap_seconds"], 30)
        self.assertEqual(post["settings"]["camera_id_mode"], "folder")
        post = experimental_postprocess(self.frames([0, 60, 121]), {"event_gap_seconds": 120})
        self.assertEqual([e["image_count"] for e in post["events"]], [3])

    def test_cameras_never_share_an_event(self):
        frames = [item("camA/1.jpg", image(time=at(0))), item("camB/1.jpg", image(time=at(1)))]
        post = experimental_postprocess(frames)
        self.assertEqual(len(post["events"]), 2)

    def test_untimed_images_group_by_sequence_number(self):
        frames = [item(f"camA/IMG_{n:04d}.JPG", image(time=None, seq=n)) for n in (1, 2, 3, 20)]
        post = experimental_postprocess(frames)
        self.assertEqual(sorted(e["image_count"] for e in post["events"]), [1, 3])

    def test_empty_frames_are_assessed_zero(self):
        frames = self.frames([0])
        post = experimental_postprocess(frames)
        combined = frames[0]["result"]["combined"]
        self.assertEqual((combined["adults"], combined["children"], combined["age_unknown"]), (0, 0, 0))
        self.assertEqual(combined["direction_counts"], dict.fromkeys(TRAVEL_DIRECTIONS, 0))
        # v2: a camera group without people is still fitted (fit_camera([])) and
        # reports "insufficient" instead of a missing (None) status.
        self.assertEqual(frames[0]["result"]["post"]["camera_calibration_status"], "insufficient")
        calibration = post["calibrations"]["camA|Acme TC-1@2000x1200"]
        self.assertEqual((calibration["status"], calibration["n"], calibration["reasons"]),
                         ("insufficient", 0, ["fewer_complete_people_than_minimum"]))


class CameraIdModeTests(unittest.TestCase):
    def run_mode(self, settings):
        frames = [item("site/cam1/IMG_0001.JPG", image(time=at(0))), item("IMG_0002.JPG", image(time=at(0)))]
        experimental_postprocess(frames, settings)
        return [f["result"]["post"]["camera_id"] for f in frames]

    def test_modes(self):
        self.assertEqual(self.run_mode(None), ["site/cam1", "."])
        self.assertEqual(self.run_mode({"camera_id_mode": "single"}), ["all", "all"])
        self.assertEqual(self.run_mode({"camera_id_mode": "regex", "camera_id_pattern": r"(?P<camera>cam\d+)"}),
                         ["cam1", "."])  # no match falls back to the folder


class CalibrationGroupingTests(unittest.TestCase):
    def test_groups_by_camera_model_and_size(self):
        person = [walker([500, 400, 600, 700])]
        frames = [item("camA/1.jpg", image(person, at(0))),
                  item("camA/2.jpg", image(person, at(1000), make="Bushnell", model="X")),
                  item("camA/3.jpg", image(person, at(2000), make=None, model=None)),
                  item("camA/4.jpg", image(person, at(3000), width=1920, height=1080)),
                  item("camB/1.jpg", image(person, at(0)))]
        post = experimental_postprocess(frames)
        self.assertEqual(set(post["calibrations"]), {
            "camA|Acme TC-1@2000x1200", "camA|Bushnell X@2000x1200", "camA|unknown@2000x1200",
            "camA|Acme TC-1@1920x1080", "camB|Acme TC-1@2000x1200"})
        for calibration in post["calibrations"].values():
            self.assertEqual((calibration["status"], calibration["n"]), ("insufficient", 1))


class RelativeHeightAgeTests(unittest.TestCase):
    def test_adult_child_and_unknown_counts(self):
        result = image([walker([300, 300, 400, 900], colour=1),            # tall, complete
                        walker([600, 540, 700, 900], colour=3),            # 0.6 of the tall person, same ground row
                        walker([1000, 300, 1100, 900], colour=5, pose=False)], at(0))  # no keypoints
        frame = item("camA/1.jpg", result)
        experimental_postprocess([frame])
        combined = result["combined"]
        self.assertEqual((combined["adults"], combined["children"], combined["age_unknown"]), (0, 1, 2))
        self.assertEqual((combined["age_by_relative_height"], combined["age_by_camera_calibration"]), (1, 0))
        self.assertEqual(combined["age_status"], "geometry_estimate")
        ages = {p["xyxy"][0]: (p["combined"]["age"], p["combined"]["age_method"]) for p in accepted(result)}
        self.assertEqual(ages, {300: ("unknown", "relative_height"), 600: ("child", "relative_height"),
                                1000: ("unknown", "none")})
        short = next(p for p in accepted(result) if p["xyxy"][0] == 600)
        self.assertAlmostEqual(short["age_evidence"]["ratio"], 0.6, places=2)
        event = experimental_postprocess_copy([frame])[1]["events"][0]
        self.assertEqual((event["adults"], event["children"], event["age_unknown"]), (0, 1, 2))


class CameraCalibrationAgeTests(unittest.TestCase):
    """24 adults whose height grows linearly with foot row, then one child."""

    @classmethod
    def setUpClass(cls):
        def box(bottom, scale=1.0):
            height = 0.4 * bottom * scale
            return [900, bottom - height, 900 + height / 3, bottom]

        cls.adults = [item(f"calib/IMG_{n:04d}.JPG", image([walker(box(500 + 25 * n), colour=n)], at(1000 * n), n))
                      for n in range(24)]
        cls.child = item("calib/IMG_0100.JPG", image([walker(box(800, 0.6), colour=99)], at(1000 * 30), 100))
        cls.other = item("elsewhere/IMG_0001.JPG", image([walker(box(800, 0.6), colour=99)], at(0), 1))
        cls.post = experimental_postprocess(cls.adults + [cls.child, cls.other])

    def test_calibration_fitted_for_the_camera(self):
        calibration = self.post["calibrations"]["calib|Acme TC-1@2000x1200"]
        self.assertEqual(calibration["status"], "ok", calibration)
        self.assertEqual(calibration["n"], 25)
        self.assertEqual(self.child["result"]["post"]["camera_calibration_status"], "ok")
        self.assertEqual(self.post["calibrations"]["elsewhere|Acme TC-1@2000x1200"]["status"], "insufficient")

    def test_child_classified_by_camera_calibration(self):
        combined = self.child["result"]["combined"]
        self.assertEqual((combined["adults"], combined["children"], combined["age_unknown"]), (0, 1, 0))
        self.assertEqual(combined["age_by_camera_calibration"], 1)
        person = accepted(self.child["result"])[0]
        self.assertEqual(person["age_evidence"]["method"], "camera_calibration")
        # The child cut is 0.82 (formerly 0.78); this child measures about 0.6.
        self.assertLessEqual(person["age_evidence"]["ratio"], AGE_POLICY["calibration_child_max"])
        self.assertAlmostEqual(person["age_evidence"]["ratio"], 0.6, delta=0.05)

    def test_adults_classified_by_camera_calibration(self):
        for frame in self.adults:
            combined = frame["result"]["combined"]
            self.assertEqual((combined["adults"], combined["children"], combined["age_by_camera_calibration"]),
                             (1, 0, 1), frame["relative_path"])

    def test_same_person_on_uncalibrated_camera_is_unknown(self):
        combined = self.other["result"]["combined"]
        self.assertEqual((combined["adults"], combined["children"], combined["age_unknown"]), (0, 0, 1))


class FailureIsolationTests(unittest.TestCase):
    def test_fuse_error_marks_record_without_stopping_others(self):
        good = item("camA/1.jpg", image([walker([500, 400, 600, 700])], at(0)))
        broken_result = image([walker([500, 400, 600, 700])], at(1))
        del broken_result["persons"]
        broken = item("camA/2.jpg", broken_result)
        failed = item("camA/3.jpg", {"status": "error", "error": "OSError: unreadable"})
        partial = item("camA/4.jpg", image([walker([500, 400, 600, 700])], at(2), status="partial_error"))
        post = experimental_postprocess([good, broken, failed, partial])
        self.assertEqual(broken_result["status"], "error")
        # v2: the message names the stage: "Post-processing failed: <stage>: <Type>: <msg>".
        self.assertTrue(broken_result["error"].startswith("Post-processing failed: fusion: KeyError"),
                        broken_result["error"])
        self.assertTrue(broken_result["postprocess_error"].startswith("fusion: KeyError"))
        self.assertEqual(broken_result["error"], "Post-processing failed: " + broken_result["postprocess_error"])
        self.assertNotIn("post", broken_result)
        # Changed: experimental_prepare() now always records its outcome (False for non-ok records
        # and fusion failures), so callers can test item["prepared"] safely.
        self.assertEqual([f["prepared"] for f in (good, broken, failed, partial)], [True, False, False, True])
        self.assertEqual(failed["result"], {"status": "error", "error": "OSError: unreadable"})
        self.assertEqual(good["result"]["status"], "ok")
        self.assertEqual(partial["result"]["status"], "partial_error")
        self.assertEqual(len(post["events"]), 1)
        self.assertEqual(post["events"][0]["images"], [good["relative_path"], partial["relative_path"]])
        self.assertEqual(good["result"]["post"]["event_image_count"], 2)

    def test_exception_message_is_recorded(self):
        real = fusion.fuse
        victim = image(time=at(1))

        def flaky(result, *args, **kwargs):
            if result is victim:
                raise RuntimeError("boom")
            return real(result, *args, **kwargs)

        frames = [item("camA/1.jpg", image(time=at(0))), item("camA/2.jpg", victim)]
        with patch("trailcam.postprocess.fuse", side_effect=flaky):
            post = experimental_postprocess(frames)
        self.assertEqual(victim["status"], "error")
        self.assertEqual(victim["postprocess_error"], "fusion: RuntimeError: boom")  # v2: stage prefix
        self.assertEqual(victim["error"], "Post-processing failed: fusion: RuntimeError: boom")
        self.assertEqual(len(post["events"]), 1)
        self.assertIn("post", frames[0]["result"])

    def test_geometry_error_is_recorded_with_its_stage(self):
        from trailcam import age_geometry
        real = age_geometry.person_geometry

        def fragile(person):
            if person["image_id"] == "camA/2.jpg":
                raise ValueError("bad keypoints")
            return real(person)

        frames = [item("camA/1.jpg", image([walker([500, 400, 600, 700])], at(0))),
                  item("camA/2.jpg", image([walker([500, 400, 600, 700])], at(1)))]
        with patch("trailcam.age_geometry.person_geometry", side_effect=fragile):
            post = experimental_postprocess(frames)
        victim = frames[1]["result"]
        self.assertEqual((victim["status"], victim["postprocess_error"]), ("error", "geometry: ValueError: bad keypoints"))
        self.assertEqual(victim["error"], "Post-processing failed: geometry: ValueError: bad keypoints")
        self.assertNotIn("post", victim)
        self.assertEqual([e["images"] for e in post["events"]], [["camA/1.jpg"]])
        self.assertEqual(post["calibrations"]["camA|Acme TC-1@2000x1200"]["n"], 1)  # the failed image adds nobody

    def test_no_usable_items(self):
        post = experimental_postprocess([item("a.jpg", {"status": "error", "error": "x"})])
        self.assertEqual((post["events"], post["calibrations"]), ([], {}))
        self.assertEqual(experimental_postprocess([])["events"], [])


class PrepareAndCompactTests(unittest.TestCase):
    """v2: ``prepare`` (fuse + camera id + compact) runs per image, right after inference."""

    BOX, FAINT = [500, 400, 600, 700], [1500, 400, 1600, 700]

    def item(self):
        result = image([walker(self.BOX, colour=4, pose_orientation="side", facing="left")], at(0))
        experts = result["experts"]
        polygon = [[500.0, 400.0], [600.0, 400.0], [600.0, 700.0]]
        experts["yoloe"]["detections"][0]["mask_polygon_xy"] = polygon
        experts["yoloe"]["detections"].append(det("dog", .2, [0, 0, 50, 50], detection_index=1,
                                                  mask_polygon_xy=[[0.0, 0.0], [50.0, 50.0]]))
        experts["yolo26n"]["detections"] = [det("person", .3, self.FAINT, detection_index=0,
                                                mask_polygon_xy=[[1500.0, 400.0]]),
                                            det("bicycle", .24, [0, 900, 60, 960], detection_index=1)]
        experts["pose"]["detections"][0]["orientation_evidence"]["features"] = {"torso_height": 120.0}
        person = result["persons"][0]
        person["mask_polygon_xy"] = polygon
        result["persons"].append({
            "person_id": "cand_0002", "candidate_id": "cand_0002", "xyxy": list(self.FAINT),
            "bbox_xyxy": list(self.FAINT), "confidence": .3,
            "members": {"yolo26n": {"detection_index": 0, "confidence": .3, "xyxy": list(self.FAINT)}},
            "keypoints": None, "pose_confidence": None, "orientation_evidence": None, "mask_polygon_xy": None,
            "appearance": look(9), "attributes": {"status": "ok", "native_orientation_label": "unknown",
                                                  "backpack_presence": None}})
        return item("camA/IMG_0001.JPG", result)

    def test_prepare_fuses_assigns_camera_and_compacts(self):
        frame = self.item()
        self.assertIs(experimental_prepare(frame), True)
        result = frame["result"]
        self.assertEqual((frame["camera_id"], frame["prepared"]), ("camA", True))
        self.assertEqual(result["combined"]["counts"]["people_total"], 1)
        # Rejected candidates are dropped; their numbers stay for the CSV.
        self.assertEqual(result["candidate_counts"], {"total": 2, "rejected": 1})
        self.assertEqual([p["person_id"] for p in result["persons"]], ["cand_0001"])
        person = result["persons"][0]
        self.assertNotIn("mask_polygon_xy", person)
        self.assertNotIn("features", person["orientation_evidence"])
        self.assertIn("appearance", person)  # still needed for association
        # Detections below KEEP_CONFIDENCE go; only YOLOE detections keep their masks.
        self.assertEqual(KEEP_CONFIDENCE, 0.25)
        experts = result["experts"]
        self.assertEqual([d["label"] for d in experts["yoloe"]["detections"]], ["person"])
        self.assertEqual(experts["yoloe"]["detections"][0]["mask_polygon_xy"][0], [500.0, 400.0])
        self.assertEqual([d["confidence"] for d in experts["yolo26n"]["detections"]], [.3])
        self.assertNotIn("mask_polygon_xy", experts["yolo26n"]["detections"][0])
        self.assertNotIn("features", experts["pose"]["detections"][0]["orientation_evidence"])
        self.assertEqual({e["detections_kept_from_confidence"] for e in experts.values()}, {KEEP_CONFIDENCE})
        json.dumps(result, allow_nan=False)

    def test_prepare_is_idempotent_and_postprocess_reuses_it(self):
        frame = self.item()
        with patch("trailcam.postprocess.fuse", wraps=fusion.fuse) as fused:
            self.assertIs(experimental_prepare(frame, {"camera_id_mode": "single"}), True)
            self.assertIs(experimental_prepare(frame), True)
            post = experimental_postprocess([frame])
        self.assertEqual(fused.call_count, 1)
        self.assertEqual(frame["camera_id"], "camA")  # camera ids follow the settings of this postprocess call
        self.assertEqual(post["events"][0]["camera_id"], "camA")
        self.assertEqual(frame["result"]["candidate_counts"], {"total": 2, "rejected": 1})
        # Appearance descriptors are dropped once association is done.
        self.assertNotIn("appearance", frame["result"]["persons"][0])
        self.assertTrue(frame["result"]["persons"][0]["track_id"].startswith("camA__"))

    def test_prepare_skips_errors_and_marks_fusion_failures(self):
        # Changed: item["prepared"] is always set (False here) instead of left missing.
        for status in ("error", "skipped", None):
            with self.subTest(status=status):
                failed = item("camA/x.jpg", {"status": status, "error": "OSError: unreadable"})
                self.assertIs(experimental_prepare(failed), False)
                self.assertIs(failed["prepared"], False)
                self.assertEqual(failed["result"], {"status": status, "error": "OSError: unreadable"})
                self.assertNotIn("camera_id", failed)
        broken = self.item()
        del broken["result"]["experts"]
        self.assertIs(experimental_prepare(broken), False)
        self.assertTrue(broken["result"]["error"].startswith("Post-processing failed: fusion: KeyError"))
        self.assertIs(broken["prepared"], False)
        with patch("trailcam.postprocess.fuse") as fused:
            self.assertIs(experimental_prepare(broken), False)  # stays failed; nothing is retried within the run
        fused.assert_not_called()
        self.assertIs(broken["prepared"], False)
        ok = self.item()
        self.assertIs(experimental_prepare(ok), True)
        self.assertIs(ok["prepared"], True)


class DuplicateFileTests(unittest.TestCase):
    """v2: images are keyed by relative path; byte-identical copies share only an image_id."""

    @staticmethod
    def copy_of(relative_path, seconds=0):
        return item(relative_path, image([walker([500, 400, 600, 700], colour=4, pose_orientation="side",
                                                 facing="left")], at(seconds), 1), image_id="img_same_sha")

    def test_copies_in_two_folders_get_their_own_events_and_tracks(self):
        a, b = self.copy_of("camA/IMG_0001.JPG"), self.copy_of("camB/IMG_0001.JPG")
        post = experimental_postprocess([a, b])
        events = {e["camera_id"]: e for e in post["events"]}
        self.assertEqual(set(events), {"camA", "camB"})
        for frame, camera in ((a, "camA"), (b, "camB")):
            with self.subTest(camera=camera):
                event = events[camera]
                self.assertEqual((event["images"], event["people_unique"], event["dir_left"]),
                                 ([frame["relative_path"]], 1, 1))
                self.assertEqual((frame["result"]["post"]["event_id"], frame["result"]["post"]["event_frame"]),
                                 (event["event_id"], 1))
                person = accepted(frame["result"])[0]
                self.assertTrue(person["track_id"].startswith(event["event_id"] + "__t"))
                self.assertEqual((person["combined"]["direction"], person["combined"]["direction_source"]),
                                 ("left", "facing"))
        self.assertNotEqual(accepted(a["result"])[0]["track_id"], accepted(b["result"])[0]["track_id"])

    def test_copies_in_one_folder_are_two_frames_of_one_event(self):
        original, duplicate = self.copy_of("camA/IMG_0001.JPG"), self.copy_of("camA/IMG_0001 - Copy.JPG")
        post = experimental_postprocess([original, duplicate])
        event, = post["events"]
        self.assertEqual(event["images"], ["camA/IMG_0001 - Copy.JPG", "camA/IMG_0001.JPG"])
        self.assertEqual((event["image_count"], event["people_max_frame"], event["people_unique"]), (2, 1, 1))
        self.assertEqual([f["result"]["post"]["event_frame"] for f in (duplicate, original)], [1, 2])
        tracks = {accepted(f["result"])[0].get("track_id") for f in (original, duplicate)}
        self.assertEqual(len(tracks), 1)
        self.assertIsNotNone(next(iter(tracks)))


class FileDateReviewTests(unittest.TestCase):
    def frames(self, camera, sources, colours=(4, 4)):
        frames = []
        for n, (source, colour) in enumerate(zip(sources, colours)):
            result = image([walker([500, 400, 600, 700], colour=colour)], at(n) if source else None, n + 1)
            result["metadata"]["time_source"] = source
            frames.append(item(f"{camera}/IMG_{n + 1:04d}.JPG", result))
        return frames

    def test_multi_image_events_timed_by_file_dates_need_review(self):
        # v2: copied files share modification times, so such events may merge visits.
        frames = (self.frames("mtime", ["file_mtime", "file_mtime"]) + self.frames("mixed", ["exif_original", "file_mtime"])
                  + self.frames("untimed", [None, None]) + self.frames("exif", ["exif_original", "exif_original"])
                  + self.frames("single", ["file_mtime"]))
        events = {e["camera_id"]: e for e in experimental_postprocess(frames)["events"]}
        self.assertEqual({camera: e["image_count"] for camera, e in events.items()},
                         {"mtime": 2, "mixed": 2, "untimed": 2, "exif": 2, "single": 1})
        for camera in ("mtime", "mixed", "untimed"):
            with self.subTest(camera=camera):
                self.assertTrue(events[camera]["needs_review"])
                self.assertEqual(events[camera]["review_reasons"], ["times_from_file_dates"])
        for camera in ("exif", "single"):
            with self.subTest(camera=camera):
                self.assertFalse(events[camera]["needs_review"])
                self.assertEqual(events[camera]["review_reasons"], [])

    def test_file_date_reason_joins_other_reasons(self):
        frames = self.frames("camA", ["file_mtime", "file_mtime"], colours=(4, 90))  # two different people
        event, = experimental_postprocess(frames)["events"]
        self.assertEqual(event["review_reasons"], ["people_unique_vs_max_frame", "times_from_file_dates"])


class CopyTests(unittest.TestCase):
    def test_postprocess_copy_leaves_inputs_untouched(self):
        frames = [item(f"camA/{n}.jpg", image([walker([100 + 200 * n, 500, 200 + 200 * n, 800])], at(5 * n)))
                  for n in range(2)]
        before = copy.deepcopy(frames)
        copies, post = experimental_postprocess_copy(frames)
        self.assertEqual(frames, before)
        self.assertIn("post", copies[0]["result"])
        self.assertEqual(post["events"][0]["people_unique"], 1)

    def test_deterministic(self):
        def build():
            return [item(f"camA/{n}.jpg", image([walker([100 + 200 * n, 500, 200 + 200 * n, 800], colour=3),
                                                 walker([1500, 300, 1600, 900], colour=9)], at(5 * n), n))
                    for n in range(3)]
        first, second = build(), build()
        self.assertEqual(experimental_postprocess(first), experimental_postprocess(second))
        self.assertEqual([f["result"] for f in first], [f["result"] for f in second])


class VlmAgeTests(unittest.TestCase):
    """Optional VLM answers attached by ``__main__.add_vlm_age`` as ``result["vlm_age"]``.

    Per-person (mosaic) answers replace the geometry age (teen -> unknown,
    unclear keeps geometry); whole-photo counts go to ``vlm_age_counts`` and
    the event maxima. Four people on one foot row: A tall (geometry unknown),
    B short (geometry child), C tall (geometry unknown), D without pose (unknown).
    """

    BOXES = {"A": [300, 300, 400, 900], "B": [600, 540, 700, 900], "C": [1000, 300, 1100, 900],
             "D": [1400, 300, 1500, 900]}

    def frame(self, vlm=None, path="camA/1.jpg", time=0, seq=1):
        people = [walker(self.BOXES[name], colour=2 * n + 1, pose=name != "D") for n, name in enumerate("ABCD")]
        result = image(people, at(time), seq)
        self.ids = {name: next(p["person_id"] for p in result["persons"] if p["xyxy"] == box)
                    for name, box in self.BOXES.items()}
        if vlm is not None:
            result["vlm_age"] = vlm(self.ids) if callable(vlm) else vlm
        return item(path, result)

    def people(self, frame):
        by_box = {tuple(p["xyxy"]): p for p in accepted(frame["result"])}
        return {name: by_box[tuple(float(v) for v in box)] for name, box in self.BOXES.items()}

    def test_geometry_baseline(self):
        frame = self.frame()
        post = experimental_postprocess([frame])
        ages = {name: (p["combined"]["age"], p["combined"]["age_method"]) for name, p in self.people(frame).items()}
        self.assertEqual(ages, {"A": ("unknown", "relative_height"), "B": ("child", "relative_height"),
                                "C": ("unknown", "relative_height"), "D": ("unknown", "none")})
        combined = frame["result"]["combined"]
        # Changed: age_by_vlm is None (blank), not 0, when no per-person answer exists.
        self.assertEqual((combined["age_by_vlm"], combined["age_model"], combined["vlm_age_counts"]), (None, None, None))
        event, = post["events"]
        self.assertEqual({k: event[f"vlm_{k}_max_frame"] for k in ("adults", "teens", "children", "unclear")},
                         dict.fromkeys(("adults", "teens", "children", "unclear")))

    def test_mosaic_answers_replace_geometry(self):
        frame = self.frame(lambda ids: {"model": "gemma4:26b", "photo": None, "persons": {
            "ages": {ids["A"]: "child", ids["B"]: "teen", ids["C"]: "unclear", ids["D"]: "adult"}, "error": None}})
        post = experimental_postprocess([frame])
        people = self.people(frame)
        a, b, c, d = (people[n]["combined"] for n in "ABCD")
        self.assertEqual((a["age"], a["age_method"], a["vlm_age"], a["geometry_age"]), ("child", "vlm", "child", "unknown"))
        # A teen is not a child: unknown, but still answered by the VLM.
        self.assertEqual((b["age"], b["age_method"], b["vlm_age"], b["geometry_age"]), ("unknown", "vlm", "teen", "child"))
        # "unclear" keeps the geometry estimate and records nothing.
        self.assertEqual((c["age"], c["age_method"]), ("unknown", "relative_height"))
        self.assertNotIn("vlm_age", c)
        self.assertNotIn("geometry_age", c)
        self.assertEqual((d["age"], d["age_method"], d["geometry_age"]), ("adult", "vlm", "unknown"))
        # The geometry decision itself is kept as evidence.
        self.assertEqual(people["A"]["age_evidence"]["age"], "unknown")
        self.assertEqual(people["A"]["age_evidence"]["method"], "relative_height")
        combined = frame["result"]["combined"]
        self.assertEqual((combined["adults"], combined["children"], combined["age_unknown"]), (1, 1, 2))
        self.assertEqual((combined["age_by_vlm"], combined["age_by_relative_height"],
                          combined["age_by_camera_calibration"]), (2, 0, 0))   # the teen (unknown) is not counted
        self.assertEqual((combined["age_model"], combined["vlm_age_counts"]), ("gemma4:26b", None))
        event, = post["events"]
        self.assertEqual((event["adults"], event["children"], event["age_unknown"]), (1, 1, 2))
        json.dumps(post, allow_nan=False)

    def test_unknown_ids_and_invalid_answers_are_ignored(self):
        frame = self.frame(lambda ids: {"model": "m", "photo": None, "persons": {
            "ages": {"cand_9999": "child", ids["A"]: "elderly", ids["B"]: None, ids["D"]: "Child"}}})
        experimental_postprocess([frame])
        ages = {name: (p["combined"]["age"], p["combined"]["age_method"]) for name, p in self.people(frame).items()}
        self.assertEqual(ages, {"A": ("unknown", "relative_height"), "B": ("child", "relative_height"),
                                "C": ("unknown", "relative_height"), "D": ("unknown", "none")})
        self.assertEqual(frame["result"]["combined"]["age_by_vlm"], 0)

    def test_partial_answers_from_a_failed_request_are_used(self):
        frame = self.frame(lambda ids: {"model": "m", "photo": None, "persons": {
            "ages": {ids["D"]: "child"}, "error": "ConnectionError: reset"}})
        experimental_postprocess([frame])
        people = self.people(frame)
        self.assertEqual((people["D"]["combined"]["age"], people["D"]["combined"]["age_method"]), ("child", "vlm"))
        self.assertEqual(people["A"]["combined"]["age_method"], "relative_height")

    def test_missing_or_empty_person_answers(self):
        # Changed: age_by_vlm is None unless a per-person (mosaic) answer exists; an
        # answered mosaic with an empty ages dict counts as answered (0).
        for persons, by_vlm in ((None, None), ({}, None), ({"ages": None}, None),
                                ({"ages": {}, "error": None}, 0), ({"ages": {}, "error": "ValueError: x"}, 0)):
            with self.subTest(persons=persons):
                frame = self.frame({"model": "m", "photo": None, "persons": persons})
                experimental_postprocess([frame])
                combined = frame["result"]["combined"]
                self.assertEqual((combined["adults"], combined["children"], combined["age_by_vlm"]), (0, 1, by_vlm))
                self.assertEqual(combined["age_model"], "m")
                self.assertEqual(combined["age_by_relative_height"], 1)

    def test_answered_mosaic_for_a_photo_without_people_counts_zero(self):
        result = image([], at(0))
        result["vlm_age"] = {"model": "m", "photo": None, "persons": {"ages": {}, "error": None}, "errors": None}
        frame = item("camA/empty.jpg", result)
        experimental_postprocess([frame])
        combined = frame["result"]["combined"]
        self.assertEqual((combined["age_by_vlm"], combined["age_model"], combined["counts"]["people_total"]), (0, "m", 0))

    def test_photo_counts_do_not_change_person_ages(self):
        counts = {"people_total": 4, "adults": 1, "teens": 1, "children": 2, "unclear": 0}
        frame = self.frame({"model": "gemma4:26b", "photo": {"counts": counts, "error": None}, "persons": None})
        experimental_postprocess([frame])
        combined = frame["result"]["combined"]
        self.assertEqual(combined["vlm_age_counts"], counts)
        # Changed: photo-only answers leave age_by_vlm None (no per-person answers).
        self.assertEqual((combined["adults"], combined["children"], combined["age_by_vlm"]), (0, 1, None))

    def test_photo_counts_give_event_maxima(self):
        answers = [{"people_total": 3, "adults": 1, "teens": 0, "children": 2, "unclear": 0},
                   {"people_total": 5, "adults": 3, "teens": 1, "children": 0, "unclear": 1},
                   None]  # the third photo's request failed
        frames = [self.frame({"model": "m", "photo": {"counts": c, "error": None if c else "ValueError: x"},
                              "persons": None}, path=f"camA/{n}.jpg", time=5 * n, seq=n)
                  for n, c in enumerate(answers)]
        other = self.frame(None, path="camB/1.jpg")   # another camera without VLM answers
        post = experimental_postprocess(frames + [other])
        event = next(e for e in post["events"] if e["camera_id"] == "camA")
        self.assertEqual(event["image_count"], 3)
        self.assertEqual({k: event[f"vlm_{k}_max_frame"] for k in ("adults", "teens", "children", "unclear")},
                         {"adults": 3, "teens": 1, "children": 2, "unclear": 1})
        self.assertIsNone(frames[2]["result"]["combined"]["vlm_age_counts"])
        quiet = next(e for e in post["events"] if e["camera_id"] == "camB")
        self.assertIsNone(quiet["vlm_children_max_frame"])

    def test_mosaic_and_photo_together(self):
        counts = {"people_total": 4, "adults": 2, "teens": 0, "children": 2, "unclear": 0}
        frame = self.frame(lambda ids: {"model": "m", "photo": {"counts": counts, "error": None},
                                        "persons": {"ages": {ids["C"]: "child"}, "error": None}})
        post = experimental_postprocess([frame])
        combined = frame["result"]["combined"]
        self.assertEqual((combined["adults"], combined["children"], combined["age_by_vlm"]), (0, 2, 1))
        self.assertEqual(combined["vlm_age_counts"], counts)
        self.assertEqual(post["events"][0]["vlm_children_max_frame"], 2)


class ConservativeDefaultAndTrackingIntegrationTests(unittest.TestCase):
    def test_public_defaults_keep_age_unknown_events_off_and_large_bags_unassessed(self):
        frames = [item(f"camera/{i}.jpg", image([
            walker([300, 300, 400, 900], colour=1),
            walker([600, 540, 700, 900], colour=3)], at(i), i)) for i in range(2)]
        result = public_postprocess(frames)
        self.assertEqual(result["settings"], DEFAULTS)
        self.assertEqual((result["events"], result["calibrations"]), ([], {}))
        for frame in frames:
            combined = frame["result"]["combined"]
            self.assertEqual((combined["adults"], combined["children"], combined["age_unknown"]), (0, 0, 2))
            self.assertEqual(combined["age_status"], "unknown")
            self.assertIsNone(combined["large_bags"])
            self.assertIsNone(combined["large_bags_uncertain"])
            self.assertEqual(combined["people_near"], combined["counts"]["people_total"])
            self.assertNotIn("event_id", frame["result"]["post"])
            self.assertTrue(all("track_id" not in p for p in accepted(frame["result"])))

    def test_missing_appearance_marks_people_and_preserves_each_person_facing(self):
        frames = [item(f"cam/{i}.jpg", image([walker([300 + 60 * i, 400, 400 + 60 * i, 700],
                       orientation=orientation)], at(i), i)) for i, orientation in enumerate(("front", "back"))]
        for frame in frames:
            for person in frame["result"]["persons"]:
                person["appearance"] = None
        result = public_postprocess(frames, {"events": True})
        self.assertEqual(len(result["events"]), 1)
        self.assertIn("missing_appearance", result["events"][0]["review_reasons"])
        self.assertEqual(result["events"][0]["direction_from_motion"], 0)
        for frame, direction in zip(frames, ("toward", "away")):
            person = accepted(frame["result"])[0]
            self.assertTrue(person["track_ambiguous"])
            self.assertIn("missing_appearance", person["track_review_reasons"])
            self.assertEqual((person["combined"]["direction"], person["combined"]["direction_source"]),
                             (direction, "facing"))
            self.assertTrue(frame["result"]["needs_review"])

    def test_backpack_counts_reach_events_with_large_bag_experiment_disabled(self):
        frames = []
        for i, count in enumerate((1, 3, 2)):
            result = image(time=at(i))
            for expert in ("yoloe", "yolo26n"):
                result["experts"][expert]["detections"].extend(
                    det("backpack", .9, [100 + 70 * n, 100, 140 + 70 * n, 200]) for n in range(count))
            frames.append(item(f"cam/{i}.jpg", result))
        result = public_postprocess(frames, {"events": True})
        self.assertEqual([f["result"]["combined"]["counts"]["backpacks"] for f in frames], [1, 3, 2])
        self.assertEqual(result["events"][0]["backpacks"], 3)
        self.assertIsNone(result["events"][0]["large_bags"])

    def test_dimensions_and_camera_metadata_are_forwarded_to_event_grouping(self):
        first = item("cam/a.jpg", image([walker([300, 300, 400, 600])], at(0)))
        resized = item("cam/b.jpg", image([walker([450, 450, 600, 900])], at(1),
                                         width=3000, height=1800))
        other_model = item("cam/c.jpg", image([walker([450, 450, 600, 900])], at(2),
                                              width=3000, height=1800, model="Other"))
        result = public_postprocess([first, resized, other_model], {"events": True})
        self.assertEqual([event["image_count"] for event in result["events"]], [1, 1, 1])
        self.assertTrue(all(event["direction_from_motion"] == 0 for event in result["events"]))


if __name__ == "__main__":
    unittest.main()
