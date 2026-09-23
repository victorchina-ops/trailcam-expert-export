"""Synthetic invariants for events, association and motion direction (no models, no photos)."""
from __future__ import annotations

import contextlib
import copy
from datetime import datetime, timedelta
import importlib.util
import json
import math
import random
import sys
import time
import types
import unittest
from unittest.mock import patch

from trailcam import events as events_module
from trailcam.events import (FACINGS, OBJECT_FIELDS, POLICY, SUMMARY_FIELDS, appearance_distance, associate,
                             event_id_for, group_events, link_cost, parse_time, summarize_event,
                             track_age, track_direction, track_observations)

HAS_SCIPY = importlib.util.find_spec("scipy") is not None


def at(seconds):
    return (datetime(2024, 5, 1, 10, 0, 0) + timedelta(seconds=seconds)).isoformat()


def hist(index, spread=None):
    values = [0.0] * 256
    for i, weight in (spread or {index: 1.0}).items():
        values[i] = weight
    return values


def look(color):
    """One-hot descriptor: identical colours -> distance 0, colours >= 2 apart -> 1."""
    return {"version": "hsv_v1", "upper": hist(color), "lower": hist(color + 1), "valid": True}


def person(pid, x, height=150.0, y=400.0, appearance=None, facing="unclear", age="unknown", large_bag=None):
    return {"person_id": pid, "box": [x - 20, y - height, x + 20, y], "foot": [x, y],
            "height_px": height, "appearance": appearance, "age": age, "facing": facing,
            "large_bag": large_bag}


def frame(image_id, time=None, seq=None, people=(), camera="cam1", objects=None, path=None):
    return {"image_id": image_id, "camera_id": camera, "capture_time": time, "sequence_number": seq,
            "relative_path": path or f"{camera}/{image_id}.jpg", "width": 1920, "height": 1080,
            "people": list(people), "objects": {} if objects is None else objects}


def no_scipy():
    return patch.dict(sys.modules, {"scipy": None, "scipy.optimize": None})


def matchers():
    """Run a scenario through the Hungarian path (when available) and the greedy fallback."""
    return ([("hungarian", contextlib.nullcontext)] if HAS_SCIPY else []) + [("greedy", no_scipy)]


def constant_distance(value):
    return lambda a, b: value


def walkers():
    """Burst of 3 frames, 1 s apart: two differently dressed people walking right."""
    return [frame(f"img{k}", at(k), 100 + k, [person("a", 100 + 60 * k, appearance=look(10)),
                                             person("b", 500 + 60 * k, appearance=look(50))])
            for k in range(3)]


def plain(value):
    if isinstance(value, dict):
        return all(isinstance(k, str) and plain(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(plain(v) for v in value)
    return type(value) in (str, int, float, bool, type(None))


class ParseTimeTests(unittest.TestCase):
    def test_iso_forms_and_malformed_values(self):
        self.assertEqual(parse_time("2024-05-01T10:00:00"), datetime(2024, 5, 1, 10))
        self.assertEqual(parse_time("2024-05-01T10:00:00.250000").microsecond, 250000)
        self.assertEqual(parse_time("2024-05-01T12:00:00+02:00"), datetime(2024, 5, 1, 10))
        for value in (None, "", "   ", "yesterday", "2024:05:01 10:00:00", 1714557600, ["2024"]):
            self.assertIsNone(parse_time(value))


class GroupEventsTests(unittest.TestCase):
    def test_two_interleaved_cameras_stay_separate(self):
        records = [frame(f"{cam}{k}", at(10 * k), k, camera=cam) for k in range(4) for cam in ("north", "south")]
        events = group_events(records)
        self.assertEqual(len(events), 2)
        self.assertEqual([[r["image_id"] for r in e] for e in events],
                         [["north0", "north1", "north2", "north3"], ["south0", "south1", "south2", "south3"]])
        for event in events:
            self.assertEqual(len({r["camera_id"] for r in event}), 1)
            self.assertEqual(len({r["event_id"] for r in event}), 1)
        self.assertTrue(events[0][0]["event_id"].startswith("north__2024-05-01T10-00-00"))
        self.assertNotEqual(events[0][0]["event_id"], events[1][0]["event_id"])

    def test_gap_splitting_boundary_is_inclusive(self):
        records = [frame("a", at(0)), frame("b", at(120)), frame("c", at(240.5)), frame("d", at(250))]
        self.assertEqual([[r["image_id"] for r in e] for e in group_events(records)], [["a", "b"], ["c", "d"]])
        self.assertEqual(len(group_events(records, gap_seconds=5)), 4)
        self.assertEqual(len(group_events(records, gap_seconds=1000)), 1)

    def test_zero_gap_keeps_only_simultaneous_images(self):
        records = [frame("a", at(0), path="x/a.jpg"), frame("b", at(0), path="x/b.jpg"), frame("c", at(1))]
        self.assertEqual([len(e) for e in group_events(records, gap_seconds=0)], [2, 1])

    def test_missing_times_fall_back_to_sequence_numbers(self):
        records = [frame("s10", seq=10), frame("s13", seq=13), frame("s17", seq=17), frame("s18", seq=18)]
        self.assertEqual([[r["image_id"] for r in e] for e in group_events(records)],
                         [["s10", "s13"], ["s17", "s18"]])

    def test_time_missing_for_one_neighbour_uses_sequence(self):
        records = [frame("timed", at(0), 41), frame("untimed", None, 42), frame("far", None, 90)]
        self.assertEqual([[r["image_id"] for r in e] for e in group_events(records)],
                         [["timed", "untimed"], ["far"]])

    def test_no_time_and_no_sequence_starts_new_events(self):
        records = [frame("a"), frame("b"), frame("c", seq=5)]
        self.assertEqual(len(group_events(records)), 3)

    def test_malformed_time_is_treated_as_missing(self):
        records = [frame("a", "not a time", 7), frame("b", "2024-13-45T99:00:00", 8)]
        self.assertEqual(len(group_events(records)), 1)

    def test_sort_order_time_then_sequence_then_path(self):
        records = [frame("late", at(30), 1), frame("tie_b", at(0), 5, path="cam1/b.jpg"),
                   frame("tie_a", at(0), 5, path="cam1/A.jpg"), frame("seq4", at(0), 4),
                   frame("untimed", None, 2)]
        (event,) = group_events(records, gap_seconds=60)
        self.assertEqual([r["image_id"] for r in event], ["seq4", "tie_a", "tie_b", "late", "untimed"])

    def test_timezone_aware_times_are_compared_in_utc(self):
        records = [frame("a", "2024-05-01T12:00:00+02:00"), frame("b", "2024-05-01T10:01:00+00:00")]
        self.assertEqual(len(group_events(records)), 1)

    def test_event_ids_are_sanitised_unique_and_use_filename_without_time(self):
        # v2: ids no longer carry a global running index (``__0001`` ...); they
        # depend only on the event itself (camera + first time or filename).
        records = [frame("r1", camera=".", seq=1, path="IMG_0001.JPG"),
                   frame("r2", camera="a/b c", seq=1, path="a/b c/IMAG 0571.jpg"),
                   frame("r3", at(0), camera="a_b_c")]
        ids = [e[0]["event_id"] for e in group_events(records)]
        self.assertEqual(ids, ["root__IMG_0001.JPG", "a_b_c__IMAG_0571.jpg", "a_b_c__2024-05-01T10-00-00"])
        self.assertEqual(event_id_for([frame("x", at(3.5))]), "cam1__2024-05-01T10-00-03")
        self.assertEqual(event_id_for([]), "root__event")
        self.assertEqual(event_id_for([frame("x", at(3.5))], index=7), "cam1__2024-05-01T10-00-03__0007")

    def test_colliding_event_ids_get_a_suffix_in_event_order(self):
        # "a/b c" and "a_b_c" sanitise to the same camera text; three events share one base id.
        records = [frame("x1", at(0), camera="a/b c"), frame("x2", at(0), camera="a_b_c"),
                   frame("x3", at(0), camera="a b/c")]
        events = group_events(records)
        self.assertEqual([[r["image_id"] for r in e] for e in events], [["x3"], ["x1"], ["x2"]])  # camera order
        base = "a_b_c__2024-05-01T10-00-00"
        self.assertEqual([e[0]["event_id"] for e in events], [base, base + "__02", base + "__03"])
        self.assertEqual(group_events(list(reversed(records))), events)  # deterministic

    def test_event_ids_are_stable_when_unrelated_photos_are_added(self):
        records = [frame("a0", at(0)), frame("a1", at(10)), frame("a2", at(900)),
                   frame("b0", at(5), camera="cam2"), frame("u0", None, 7, path="cam1/IMG_0007.JPG")]
        before = {r["image_id"]: r["event_id"] for e in group_events(records) for r in e}
        additions = [frame("new_camera", at(-3600), camera="cam0"),   # sorts before every camera
                     frame("earlier", at(-600)),                      # earlier event on the same camera
                     frame("later", at(5000)), frame("untimed", None, 99, path="cam1/IMG_0099.JPG")]
        after = {r["image_id"]: r["event_id"] for e in group_events(additions + records) for r in e}
        self.assertEqual({k: after[k] for k in before}, before)
        self.assertEqual(before["a0"], "cam1__2024-05-01T10-00-00")
        self.assertEqual(before["a0"], before["a1"])
        self.assertEqual(before["u0"], "cam1__IMG_0007.JPG")
        self.assertEqual((len(set(before.values())), len(set(after.values()))), (4, 8))

    def test_inputs_are_not_mutated_and_empty_input(self):
        records = walkers()
        before = copy.deepcopy(records)
        events = group_events(records)
        self.assertEqual(records, before)
        self.assertNotIn("event_id", records[0])
        self.assertIsNot(events[0][0], records[0])
        self.assertEqual(group_events([]), [])

    def test_invalid_gap_raises(self):
        for gap in (-1, None, float("nan"), "120"):
            with self.assertRaises(ValueError):
                group_events([frame("a", at(0))], gap_seconds=gap)


class AppearanceTests(unittest.TestCase):
    def test_local_fallback_is_mean_hellinger_of_usable_parts(self):
        with patch.dict(sys.modules, {"trailcam.appearance": None}):
            self.assertEqual(appearance_distance(look(10), look(10)), 0.0)
            self.assertAlmostEqual(appearance_distance(look(10), look(50)), 1.0)
            half = {"version": "hsv_v1", "upper": hist(0, {10: .5, 11: .5}), "lower": hist(11), "valid": True}
            self.assertAlmostEqual(appearance_distance(look(10), half), math.sqrt(1 - math.sqrt(.5)) / 2, places=6)
            upper_only = {"version": "hsv_v1", "upper": hist(10), "lower": [0.0] * 256, "valid": True}
            self.assertEqual(appearance_distance(look(10), upper_only), 0.0)
            self.assertIsNone(appearance_distance(look(10), {**look(10), "valid": False}))
            self.assertIsNone(appearance_distance(look(10), None))
            self.assertIsNone(appearance_distance(look(10), {"upper": [1.0], "lower": [], "valid": True}))

    def test_delegates_to_appearance_module_and_clamps(self):
        fake = types.ModuleType("trailcam.appearance")
        fake.distance = lambda a, b: 0.25
        with patch.dict(sys.modules, {"trailcam.appearance": fake}):
            self.assertEqual(appearance_distance(look(1), look(9)), 0.25)
        fake.distance = lambda a, b: 1.7
        with patch.dict(sys.modules, {"trailcam.appearance": fake}):
            self.assertEqual(appearance_distance(look(1), look(9)), 1.0)

        def broken(a, b):
            raise KeyError("upper")
        fake.distance = broken
        with patch.dict(sys.modules, {"trailcam.appearance": fake}):
            self.assertIsNone(appearance_distance({"junk": 1}, look(9)))


class LinkCostTests(unittest.TestCase):
    def test_identical_observations_cost_nothing(self):
        cost, passed = link_cost(person("a", 100, appearance=look(10)), person("b", 100, appearance=look(10)), 1.0)
        self.assertAlmostEqual(cost, 0.0, places=4)
        self.assertTrue(passed)

    def test_unknown_terms_cost_half(self):
        self.assertAlmostEqual(link_cost(person("a", 100), person("b", 100), 1.0, constant_distance(None))[0], .30)
        blank = {"person_id": "x", "box": [5, 5, 5, 5], "height_px": 0, "foot": None, "appearance": None}
        cost, passed = link_cost(blank, blank, 1.0, constant_distance(None))
        self.assertAlmostEqual(cost, .50)
        self.assertTrue(passed)

    def test_box_fallback_for_height_and_foot(self):
        a = {"box": [0, 0, 40, 100], "height_px": None, "foot": None, "appearance": {}}
        b = {"box": [0, 0, 40, 150], "height_px": float("nan"), "foot": ["x", 1], "appearance": {}}
        cost, passed = link_cost(a, b, 0.0, constant_distance(0.0))
        self.assertTrue(passed)
        self.assertAlmostEqual(cost, .25 * math.log(1.5) / math.log(2) + .15 * 50 / 125)
        self.assertFalse(link_cost(a, {"box": [0, 0, 40, 200], "appearance": {}}, 0.0, constant_distance(0.0))[1])

    def test_displacement_allowance_grows_with_time(self):
        a, b = person("a", 100, appearance={}), person("b", 400, appearance={})
        near = link_cost(a, b, 0.0, constant_distance(0.0))[0]
        later = link_cost(a, b, 10.0, constant_distance(0.0))[0]
        self.assertAlmostEqual(near, .15)
        self.assertLess(later, near)


class AssociateTests(unittest.TestCase):
    def test_two_walkers_burst_is_one_event_with_two_right_tracks(self):
        for name, context in matchers():
            with self.subTest(matcher=name), context():
                events = group_events(walkers())
                self.assertEqual(len(events), 1)
                tracks = associate(events[0])
                self.assertEqual(len(tracks), 2)
                self.assertEqual({t["matcher"] for t in tracks}, {name})
                self.assertEqual([t["observations"] for t in tracks],
                                 [[("img0", "a"), ("img1", "a"), ("img2", "a")],
                                  [("img0", "b"), ("img1", "b"), ("img2", "b")]])
                self.assertFalse(any(t["ambiguous"] for t in tracks))
                self.assertTrue(tracks[0]["track_id"].startswith(events[0][0]["event_id"] + "__t"))
                for track in tracks:
                    direction = track_direction(track_observations(events[0], track))
                    self.assertEqual((direction["direction"], direction["source"]), ("right", "motion"))
                    self.assertAlmostEqual(direction["lateral"], 120 / 150)
                    self.assertEqual(direction["seconds"], 2.0)
                    self.assertEqual(track["direction"], direction)  # attached by associate()
                    self.assertEqual(track_direction(track, events[0]), direction)  # track-dict form
                    self.assertEqual(track["age"], "unknown")
                summary = summarize_event(events[0], tracks)
                self.assertEqual((summary["people_unique"], summary["people_max_frame"]), (2, 2))
                self.assertEqual((summary["dir_right"], summary["direction_from_motion"]), (2, 2))
                self.assertFalse(summary["needs_review"])

    def test_approaching_person_is_toward_and_receding_is_away(self):
        heights = (100, 130, 165)
        approach = [frame(f"i{k}", at(k), people=[person("p", 300, h, 300 + 40 * k, look(20))])
                    for k, h in enumerate(heights)]
        recede = [frame(f"i{k}", at(k), people=[person("p", 300, h, 390 - 40 * k, look(20))])
                  for k, h in enumerate(reversed(heights))]
        for records, expected in ((approach, "toward"), (recede, "away")):
            for name, context in matchers():
                with self.subTest(expected=expected, matcher=name), context():
                    tracks = associate(records)
                    self.assertEqual(len(tracks), 1)
                    direction = track_direction(track_observations(records, tracks[0]))
                    self.assertEqual((direction["direction"], direction["source"]), (expected, "motion"))
                    self.assertAlmostEqual(abs(direction["radial"]), math.log(1.65), places=6)

    def test_person_missed_in_middle_frame_is_bridged(self):
        def group(k, include_c):
            people = [person("a", 100 + 10 * k, appearance=look(10)), person("b", 300 + 10 * k, appearance=look(50))]
            if include_c:
                people.append(person("c", 500 + 10 * k, appearance=look(90)))
            return frame(f"f{k}", at(k), people=people)
        records = [group(0, True), group(1, False), group(2, True)]
        for name, context in matchers():
            with self.subTest(matcher=name), context():
                tracks = associate(records)
                self.assertEqual(len(tracks), 3)
                c_track = next(t for t in tracks if t["observations"][0] == ("f0", "c"))
                self.assertEqual(c_track["observations"], [("f0", "c"), ("f2", "c")])
                self.assertEqual(c_track["frames"], [0, 2])
                summary = summarize_event(records, tracks)
                self.assertEqual((summary["people_unique"], summary["people_max_frame"]), (3, 3))
                self.assertFalse(summary["needs_review"])
                self.assertEqual(len(associate(records, max_frame_gap=1)), 4)

    def test_appearance_dissimilar_people_are_not_merged(self):
        records = [frame("f0", at(0), people=[person("x", 300, appearance=look(10))]),
                   frame("f1", at(1), people=[person("y", 305, appearance=look(90))])]
        for name, context in matchers():
            with self.subTest(matcher=name), context():
                tracks = associate(records)
                self.assertEqual(len(tracks), 2)
                self.assertFalse(any(t["ambiguous"] for t in tracks))
                self.assertTrue(summarize_event(records, tracks)["needs_review"])  # 2 unique vs 1 per frame

    def test_appearance_gate_boundary(self):
        records = [frame("f0", at(0), people=[person("x", 300, appearance={})]),
                   frame("f1", at(1), people=[person("y", 300, appearance={})])]
        self.assertEqual(len(associate(records, distance=constant_distance(.60))), 1)
        tracks = associate(records, distance=constant_distance(.61))
        self.assertEqual(len(tracks), 2)
        self.assertFalse(tracks[1]["ambiguous"])  # hard gate failure is not a near miss

    def test_size_ratio_gate_boundary(self):
        for height, expected in ((180.0, 1), (181.0, 2)):
            records = [frame("f0", at(0), people=[person("x", 300, 100.0, appearance=look(10))]),
                       frame("f1", at(1), people=[person("y", 300, height, appearance=look(10))])]
            with self.subTest(height=height):
                self.assertEqual(len(associate(records)), expected)

    def test_cost_gate_and_near_miss_flag(self):
        records = [frame("f0", at(0), people=[person("x", 300, 100.0, appearance={})]),
                   frame("f1", at(1), people=[person("y", 300, 170.0, appearance={})])]
        linked = associate(records, distance=constant_distance(.55))  # 0.33 + 0.25*log2(1.7) = 0.521
        self.assertEqual(len(linked), 1)
        self.assertFalse(linked[0]["ambiguous"])
        split = associate(records, distance=constant_distance(.60))  # 0.36 + 0.191 = 0.551 > 0.55
        self.assertEqual(len(split), 2)
        self.assertEqual(split[1]["reasons"], ["near_gate_candidate"])
        self.assertTrue(summarize_event(records, split)["needs_review"])

    def test_unknown_appearance_links_remain_uncertain_even_without_rivals(self):
        single = [frame(f"f{k}", at(k), people=[person("p", 300 + 5 * k)]) for k in range(3)]
        tracks = associate(single, distance=constant_distance(None))
        self.assertEqual(len(tracks), 1)
        self.assertTrue(tracks[0]["ambiguous"])
        self.assertEqual(tracks[0]["reasons"], ["missing_appearance"])
        self.assertEqual(tracks[0]["direction"]["source"], "none")
        pair = [frame(f"f{k}", at(k), people=[person("p", 300 + 10 * k), person("q", 320 + 10 * k)])
                for k in range(2)]
        tracks = associate(pair, distance=constant_distance(None))
        self.assertEqual(len(tracks), 2)
        self.assertTrue(all(t["reasons"] == ["competing_candidate", "missing_appearance"] for t in tracks))
        self.assertTrue(summarize_event(pair, tracks)["needs_review"])

    @unittest.skipUnless(HAS_SCIPY, "scipy not installed")
    def test_hungarian_is_globally_optimal_where_greedy_is_not(self):
        table = {("x", "p"): .20, ("y", "p"): .25, ("x", "q"): .30, ("y", "q"): .90}
        records = [frame("f0", at(0), people=[person("x", 100, appearance="x"), person("y", 300, appearance="y")]),
                   frame("f1", at(1), people=[person("p", 100, appearance="p"), person("q", 300, appearance="q")])]

        def measure(a, b):
            return table[(a, b)]
        optimal = associate(records, distance=measure)
        self.assertEqual(len(optimal), 2)
        self.assertEqual(sorted(t["observations"] for t in optimal),
                         [[("f0", "x"), ("f1", "q")], [("f0", "y"), ("f1", "p")]])
        with no_scipy():
            greedy = associate(records, distance=measure)
        self.assertEqual(len(greedy), 3)
        self.assertEqual(greedy[0]["observations"], [("f0", "x"), ("f1", "p")])
        self.assertEqual({t["matcher"] for t in greedy}, {"greedy"})

    def test_missing_times_use_frame_steps(self):
        records = [frame(f"s{k}", None, 10 + k, [person("p", 100 + 30 * k, appearance=look(10))]) for k in range(3)]
        (event,) = group_events(records)
        tracks = associate(event)
        self.assertEqual(len(tracks), 1)
        direction = track_direction(track_observations(event, tracks[0]))
        self.assertEqual((direction["direction"], direction["source"], direction["seconds"]),
                         ("right", "motion", None))
        self.assertEqual(summarize_event(event, tracks)["start"], None)

    def test_degenerate_and_empty_inputs(self):
        self.assertEqual(associate([]), [])
        self.assertEqual(associate([frame("f0", at(0)), frame("f1", at(1))]), [])
        broken = {"person_id": "z", "box": [5, 5, 5, 5], "foot": None, "height_px": -3, "appearance": None,
                  "facing": "left"}
        records = [frame("f0", at(0), people=[broken, "not a person"]), frame("f1", at(1), people=[dict(broken)]),
                   {**frame("f2", at(2)), "people": None}]
        tracks = associate(records)
        self.assertEqual([t["observations"] for t in tracks], [[("f0", "z"), ("f1", "z")]])
        self.assertEqual(tracks[0]["direction"]["source"], "facing")
        self.assertEqual(track_direction(tracks[0])["source"], "none")  # track dict without its records
        direction = track_direction(track_observations(records, tracks[0]))
        self.assertEqual((direction["direction"], direction["source"]), ("left", "facing"))
        no_ids = [frame("f0", at(0), people=[{"box": [0, 0, 10, 50]}])]
        self.assertEqual(associate(no_ids)[0]["observations"], [("f0", "person_0001")])

    def test_invalid_max_frame_gap_raises(self):
        for gap in (0, -1, True, 1.5, "2"):
            with self.assertRaises(ValueError):
                associate(walkers(), max_frame_gap=gap)

    def test_deterministic_plain_json_output_and_inputs_untouched(self):
        records = walkers()
        before = copy.deepcopy(records)
        events = group_events(records)
        first = [(associate(e), summarize_event(e)) for e in events]
        second = [(associate(e), summarize_event(e)) for e in group_events(records)]
        self.assertEqual(first, second)
        self.assertEqual(records, before)
        self.assertTrue(plain(first))
        json.dumps(first, allow_nan=False)


class DirectionTests(unittest.TestCase):
    @staticmethod
    def obs(x, height, seconds=None, facing="unclear", y=400.0):
        return {"foot": [x, y], "height_px": height, "facing": facing,
                "capture_time": at(seconds) if seconds is not None else None}

    def test_single_observation_uses_facing(self):
        result = track_direction([self.obs(100, 150, 0, "left")])
        self.assertEqual(result, {"direction": "left", "source": "facing", "lateral": None,
                                  "radial": None, "seconds": 0.0})

    def test_facing_majority_ties_and_invalid_values(self):
        no_geometry = {"foot": None, "height_px": None}
        votes = [{**no_geometry, "facing": f} for f in ("right", "right", "left", "unclear")]
        self.assertEqual(track_direction(votes)["direction"], "right")
        self.assertEqual(track_direction(votes)["source"], "facing")
        tie = [{**no_geometry, "facing": f} for f in ("right", "left")]
        self.assertEqual(track_direction(tie)["source"], "none")
        for facing in (None, "unclear", "north"):
            result = track_direction([self.obs(1, 100, facing=facing)])
            self.assertEqual((result["direction"], result["source"]), ("unclear", "none"))
        self.assertEqual(track_direction([])["source"], "none")
        self.assertEqual(track_direction(None)["direction"], "unclear")

    def test_lateral_threshold_boundary(self):
        self.assertEqual(track_direction([self.obs(100, 100, 0), self.obs(125, 100, 1)])["direction"], "right")
        self.assertEqual(track_direction([self.obs(100, 100, 0), self.obs(75, 100, 1)])["direction"], "left")
        self.assertEqual(track_direction([self.obs(100, 100, 0), self.obs(124, 100, 1)])["direction"], "unclear")

    def test_stationary_needs_four_seconds(self):
        self.assertEqual(track_direction([self.obs(100, 100, 0), self.obs(110, 100, 4)])["direction"], "stationary")
        self.assertEqual(track_direction([self.obs(100, 100, 0), self.obs(110, 100, 3.9)])["direction"], "unclear")
        self.assertEqual(track_direction([self.obs(100, 100), self.obs(100, 100)])["direction"], "unclear")

    def test_radial_threshold_and_dominance(self):
        grow = 100 * math.exp(.2)
        self.assertEqual(track_direction([self.obs(100, 100, 0), self.obs(130, grow, 1)])["direction"], "toward")
        self.assertEqual(track_direction([self.obs(100, 100, 0), self.obs(150, grow, 1)])["direction"], "right")
        self.assertEqual(track_direction([self.obs(100, grow, 0), self.obs(100, 100, 1)])["direction"], "away")
        small = 100 * math.exp(.14)
        self.assertEqual(track_direction([self.obs(100, 100, 0), self.obs(100, small, 1)])["direction"], "unclear")

    def test_motion_uses_first_and_last_observation_only(self):
        path = [self.obs(100, 100, 0), self.obs(400, 100, 1), self.obs(110, 100, 2)]
        result = track_direction(path)
        self.assertEqual((result["direction"], result["source"]), ("unclear", "motion"))
        self.assertAlmostEqual(result["lateral"], .1)


class SummaryTests(unittest.TestCase):
    def records(self):
        return [
            frame("f0", at(0), people=[person("p1", 100, age="adult"), person("p2", 300, age="child")],
                  objects={"bicycles": 1, "dogs": 0, "strollers": None, "large_bags": 1}),
            frame("f1", at(1), people=[person("p1", 110), person("p2", 310, age="child"),
                                       person("p3", 500, age="adult", facing="left")],
                  objects={"bicycles": 2, "dogs": None, "large_bags_uncertain": 1, "atv_utv": True}),
            frame("f2", at(2), people=[person("p2", 320), person("p3", 510, age="child")],
                  objects={"bicycles": 0, "large_bags": 2, "motorcycles": -1, "other_vehicles": 1.0}),
        ]

    def tracks(self, extra=(), ambiguous=False):
        base = [{"track_id": "t1", "observations": [("f0", "p1"), ("f1", "p1")], "ambiguous": ambiguous},
                {"track_id": "t2", "observations": [("f0", "p2"), ("f1", "p2"), ("f2", "p2")], "ambiguous": False},
                {"track_id": "t3", "observations": [("f1", "p3"), ("f2", "p3")], "ambiguous": False}]
        return base + list(extra)

    def test_counts_ages_objects_and_review(self):
        records = [{**r, "event_id": "cam1__x__0001"} for r in self.records()]
        summary = summarize_event(records, self.tracks())
        self.assertEqual(tuple(summary), SUMMARY_FIELDS)
        self.assertEqual(summary["event_id"], "cam1__x__0001")
        self.assertEqual((summary["start"], summary["end"], summary["duration_seconds"]),
                         ("2024-05-01T10:00:00", "2024-05-01T10:00:02", 2.0))
        self.assertEqual((summary["image_count"], summary["images"]), (3, ["f0", "f1", "f2"]))
        self.assertEqual((summary["people_unique"], summary["people_max_frame"]), (3, 3))
        self.assertEqual((summary["adults"], summary["children"], summary["age_unknown"]), (1, 1, 1))
        self.assertEqual(summary["dir_unclear"], 3)
        self.assertEqual(summary["direction_from_motion"], 3)
        self.assertEqual({f: summary[f] for f in OBJECT_FIELDS},
                         {"bicycles": 2, "strollers": None, "motorcycles": None, "atv_utv": None,
                          "other_vehicles": 1, "dogs": 0, "backpacks": None, "large_bags": 2, "large_bags_uncertain": 1})
        self.assertFalse(summary["needs_review"])
        self.assertEqual(summary["review_reasons"], [])
        flagged = summarize_event(records, self.tracks(ambiguous=True))
        self.assertTrue(flagged["needs_review"])
        self.assertEqual(flagged["review_reasons"], ["ambiguous_track"])

    def test_review_threshold_is_more_than_half_of_busiest_frame(self):
        single = {"track_id": "t4", "observations": [("f1", "p3")]}
        self.assertFalse(summarize_event(self.records(), self.tracks([single]))["needs_review"])  # 4 vs 3
        self.assertTrue(summarize_event(self.records(), self.tracks([single, single]))["needs_review"])  # 5 vs 3

    def test_single_observation_tracks_count_facing(self):
        records = [frame("f0", at(0), people=[person("a", 100, facing="toward"), person("b", 300)])]
        summary = summarize_event(records)
        self.assertEqual((summary["dir_toward"], summary["dir_unclear"]), (1, 1))
        self.assertEqual((summary["direction_from_facing"], summary["direction_from_motion"]), (1, 0))
        self.assertEqual(summary["duration_seconds"], 0.0)

    def test_track_age_rules(self):
        self.assertEqual(track_age([{"age": "child"}, {"age": "unknown"}]), "child")
        self.assertEqual(track_age([{"age": "adult"}, {"age": None}]), "adult")
        self.assertEqual(track_age([{"age": "adult"}, {"age": {"age": "child"}}]), "unknown")
        self.assertEqual(track_age([{"age": "elder"}]), "unknown")
        self.assertEqual(track_age([]), "unknown")

    def test_unresolvable_observations_and_list_pairs_after_json(self):
        tracks = json.loads(json.dumps(self.tracks([{"track_id": "ghost", "observations": [["nope", "p9"], "bad"]}])))
        summary = summarize_event(self.records(), tracks)
        self.assertEqual(summary["people_unique"], 4)
        self.assertEqual(summary["age_unknown"], 2)

    def test_empty_and_unassessed_events(self):
        empty = summarize_event([], [])
        self.assertEqual((empty["event_id"], empty["image_count"], empty["people_unique"]), (None, 0, 0))
        self.assertFalse(empty["needs_review"])
        self.assertIsNone(empty["bicycles"])
        unassessed = [{**frame("e0", at(0), objects={"dogs": 1}), "people": None}]
        summary = summarize_event(unassessed)
        self.assertIsNone(summary["people_unique"])
        self.assertIsNone(summary["dir_left"])
        self.assertEqual(summary["dogs"], 1)
        self.assertEqual(summary["event_id"], "cam1__2024-05-01T10-00-00")
        self.assertFalse(summary["needs_review"])
        self.assertEqual(summary["review_reasons"], [])

    def test_default_tracks_are_computed(self):
        (event,) = group_events(walkers())
        self.assertEqual(summarize_event(event), summarize_event(event, associate(event)))


# ---------------------------------------------------------------------------
# Adversarial review: malformed input, extreme values, determinism, JSON, speed.

def descriptor(rng):
    """Realistic dense 256-bin descriptor (random, L1-normalised, 5 decimals)."""
    def part():
        values = [rng.random() ** 6 for _ in range(256)]
        total = sum(values)
        return [round(v / total, 5) for v in values]
    return {"version": "hsv_v1", "upper": part(), "lower": part(), "valid": True}


def crowd(seed, frames=8, people=6):
    """People walking right with individual looks and jittered size/position; some missed per frame."""
    rng = random.Random(seed)
    looks = [descriptor(rng) for _ in range(people)]
    records = []
    for k in range(frames):
        present = [n for n in range(people) if rng.random() > .2]
        records.append(frame(f"c{k:02d}", at(2 * k), 500 + k, [
            person(f"p{n}", 100 + 250 * n + 20 * k + rng.uniform(-15, 15), rng.uniform(120, 180),
                   400 + rng.uniform(-30, 30), looks[n], rng.choice(FACINGS), rng.choice(("adult", "child")))
            for n in present]))
    return records


class AdversarialGroupEventsTests(unittest.TestCase):
    def test_unicode_digit_sequence_numbers(self):
        records = [frame("sup", seq="²"), frame("arabic", seq="٣"), frame("four", seq=4)]
        self.assertEqual([[r["image_id"] for r in e] for e in group_events(records)],
                         [["arabic", "four"], ["sup"]])  # superscript two is not a sequence number

    def test_offset_times_keep_recorded_wall_clock(self):
        records = [frame("b", "2024-04-30T22:00:30+00:00"), frame("a", "2024-05-01T01:00:00+03:00")]
        (event,) = group_events(records)
        self.assertEqual([r["image_id"] for r in event], ["a", "b"])
        self.assertEqual(event[0]["event_id"], "cam1__2024-05-01T01-00-00")  # v2: no running index
        summary = summarize_event(event)
        self.assertEqual((summary["start"], summary["end"], summary["duration_seconds"]),
                         ("2024-05-01T01:00:00+03:00", "2024-04-30T22:00:30+00:00", 30.0))

    def test_extreme_and_overflowing_times(self):
        records = [frame("min", "0001-01-01T00:00:00"), frame("max", "9999-12-31T23:59:59.999999"),
                   frame("overflow", "0001-01-01T00:00:00+05:00")]
        self.assertIsNone(parse_time("0001-01-01T00:00:00+05:00"))
        events = group_events(records)
        self.assertEqual([[r["image_id"] for r in e] for e in events], [["min"], ["max"], ["overflow"]])
        self.assertEqual([e[0]["event_id"] for e in events],  # v2: no running index
                         ["cam1__0001-01-01T00-00-00", "cam1__9999-12-31T23-59-59", "cam1__overflow.jpg"])
        summaries = [summarize_event(e) for e in events]
        self.assertIsNone(summaries[2]["start"])
        json.dumps(summaries, allow_nan=False)

    def test_camera_id_types_and_missing_cameras(self):
        records = [frame("int", at(0), camera=5), frame("str", at(1), camera="5"), frame("none", at(0), camera=None)]
        events = group_events(records)
        self.assertEqual([[r["image_id"] for r in e] for e in events], [["none"], ["int", "str"]])
        self.assertTrue(events[0][0]["event_id"].startswith("root__"))

    def test_fractional_gap_boundary(self):
        records = [frame("a", at(0)), frame("b", at(0.5)), frame("c", at(1.25))]
        self.assertEqual([len(e) for e in group_events(records, gap_seconds=0.5)], [2, 1])

    def test_input_order_does_not_change_output(self):
        records = (crowd(3) + [frame("other", at(5), camera="cam2")]
                   + [frame(f"u{k}", None, 900 + k) for k in range(3)])
        expected = group_events(records)
        expected_tracks = [associate(e) for e in expected]
        rng = random.Random(7)
        for _ in range(3):
            shuffled = records[:]
            rng.shuffle(shuffled)
            got = group_events(shuffled)
            self.assertEqual(got, expected)
            self.assertEqual([associate(e) for e in got], expected_tracks)

    def test_large_inventory_is_fast(self):
        records = [frame(f"n{c}_{k}", at(100 * k + 100 * (k // 5) + c), k, camera=f"cam{c}")
                   for c in range(10) for k in range(2000)]
        start = time.perf_counter()
        events = group_events(records)
        self.assertLess(time.perf_counter() - start, 5.0)
        self.assertEqual(sum(len(e) for e in events), 20000)
        self.assertEqual(len(events), 10 * 400)  # every 5th step is a 200 s gap
        self.assertEqual(len({e[0]["event_id"] for e in events}), len(events))


class AdversarialAppearanceTests(unittest.TestCase):
    def test_underflowing_histograms_do_not_crash(self):
        tiny = {"version": "hsv_v1", "valid": True, "upper": [1e-200] + [0.0] * 255, "lower": None}
        self.assertIn(appearance_distance(tiny, tiny), (None, 0.0))  # never raises
        with patch.dict(sys.modules, {"trailcam.appearance": None}):
            self.assertEqual(appearance_distance(tiny, tiny), 0.0)  # fallback normalises first
        fake = types.ModuleType("trailcam.appearance")

        def divide(a, b):
            raise ZeroDivisionError("float division by zero")
        fake.distance = divide
        with patch.dict(sys.modules, {"trailcam.appearance": fake}):
            self.assertIsNone(appearance_distance(look(1), look(1)))
        records = [frame(f"f{k}", at(k), people=[person("p", 100, appearance=tiny)]) for k in range(2)]
        self.assertEqual(len(associate(records)), 1)

    def test_local_fallback_rejects_malformed_descriptors(self):
        bad = [{**look(1), "version": "hsv_v2"}, {**look(1), "valid": "yes"},
               {**look(1), "upper": hist(1)[:255], "lower": None},
               {**look(1), "upper": [float("nan")] + [0.0] * 255, "lower": None},
               {**look(1), "upper": [-1.0, 2.0] + [0.0] * 254, "lower": None},
               {**look(1), "upper": [1e308] * 256, "lower": None},
               {**look(1), "upper": ["x"] * 256, "lower": None}, [], "look"]
        with patch.dict(sys.modules, {"trailcam.appearance": None}):
            for value in bad:
                with self.subTest(value=str(value)[:40]):
                    self.assertIsNone(appearance_distance(look(1), value))

    def test_local_fallback_agrees_with_appearance_module(self):
        rng = random.Random(5)
        pairs = [(descriptor(rng), descriptor(rng)) for _ in range(4)]
        pairs += [(look(3), look(3)), (look(3), look(40)), ({**look(3), "upper": None}, look(3))]
        real = [appearance_distance(a, b) for a, b in pairs]
        with patch.dict(sys.modules, {"trailcam.appearance": None}):
            local = [appearance_distance(a, b) for a, b in pairs]
        for x, y in zip(real, local):
            self.assertAlmostEqual(x, y, places=5)


class AdversarialLinkCostTests(unittest.TestCase):
    def test_extreme_geometry_stays_finite(self):
        cases = [(1e308, 1e308, [0, 0], [1e308, 1e308]), (1e-310, 1e-310, [0, 0], [1, 0]),
                 (1e300, 1e-300, [-1e308, 0], [1e308, 0]), (150, 150, [0, 0], [0, 0])]
        for ha, hb, fa, fb in cases:
            for seconds in (0.0, 1.0, float("nan"), float("inf"), -5, None, "3"):
                with self.subTest(ha=ha, hb=hb, seconds=seconds):
                    cost, passed = link_cost({"height_px": ha, "foot": fa, "appearance": None},
                                             {"height_px": hb, "foot": fb, "appearance": None},
                                             seconds, constant_distance(None))
                    self.assertTrue(math.isfinite(cost) and 0 <= cost <= 1)
                    self.assertIsInstance(passed, bool)

    def test_degenerate_boxes_mean_unknown_geometry(self):
        for box in ([10, 10, 10, 50], [10, 50, 20, 10], [0, 0, 0, 0], [float("nan"), 0, 5, 5],
                    [True, 0, 5, 5], "0 0 5 5", [1, 2, 3], None, [-1e308, -1e308, 1e308, 1e308]):
            with self.subTest(box=box):
                blank = {"box": box, "height_px": None, "foot": None, "appearance": None}
                self.assertEqual(link_cost(blank, dict(blank), 1.0, constant_distance(None)), (.5, True))

    def test_pruned_links_are_never_valid_or_near_gate(self):
        rng = random.Random(11)
        near = POLICY["max_cost"] + POLICY["ambiguity_margin"]
        pruned_count = 0
        for _ in range(400):
            a = person("a", rng.uniform(0, 1900), rng.uniform(20, 400), rng.uniform(200, 1000), appearance={})
            b = person("b", rng.uniform(0, 1900), rng.uniform(20, 400), rng.uniform(200, 1000), appearance={})
            measure = constant_distance(rng.choice([None, 0.0, 0.3, 0.6, 0.9]))
            seconds = rng.uniform(0, 5)
            full = link_cost(a, b, seconds, measure)
            pruned = events_module._link(a, b, seconds, measure, near)
            if pruned != full:
                pruned_count += 1
                self.assertFalse(full[1] and full[0] <= near)
                self.assertLessEqual(pruned[0], full[0] + 1e-12)
        self.assertGreater(pruned_count, 0)

    def test_appearance_distance_skipped_for_size_gated_pairs(self):
        calls = []

        def counting(a, b):
            calls.append((a, b))
            return 0.0
        small, tall = person("small", 100, 60.0, appearance={}), person("tall", 100, 200.0, appearance={})
        records = [frame("f0", at(0), people=[small]), frame("f1", at(1), people=[tall])]
        self.assertEqual(len(associate(records, distance=counting)), 2)
        self.assertEqual(calls, [])
        self.assertFalse(link_cost(small, tall, 1.0, counting)[1])
        self.assertEqual(len(calls), 1)  # the public link_cost still computes the full cost


class AdversarialAssociateTests(unittest.TestCase):
    def test_one_strong_link_is_not_traded_for_two_weak_ones(self):
        table = {("a", "x"): .05, ("a", "y"): .55, ("b", "x"): .55, ("b", "y"): .99}
        records = [frame("f0", at(0), people=[person("a", 100, appearance="a"), person("b", 300, appearance="b")]),
                   frame("f1", at(1), people=[person("x", 100, appearance="x"), person("y", 300, appearance="y")])]
        for name, context in matchers():
            with self.subTest(matcher=name), context():
                tracks = associate(records, distance=lambda p, q: table[(p, q)])
                self.assertEqual([t["observations"] for t in tracks],
                                 [[("f0", "a"), ("f1", "x")], [("f0", "b")], [("f1", "y")]])
                self.assertEqual(tracks[0]["direction"]["direction"], "unclear")

    def test_link_exactly_at_the_cost_gate_is_kept(self):
        records = [frame("f0", at(0), people=[person("b", 900, 60.0, appearance={}), person("a", 300, appearance={})]),
                   frame("f1", at(1), people=[person("y", 300, appearance={})])]
        cost = link_cost(records[0]["people"][1], records[1]["people"][0], 1.0, constant_distance(.5))[0]
        for name, context in matchers():
            with self.subTest(matcher=name), context(), patch.dict(POLICY, max_cost=cost):
                tracks = associate(records, distance=constant_distance(.5))
                self.assertEqual([t["observations"] for t in tracks],
                                 [[("f0", "b")], [("f0", "a"), ("f1", "y")]])
        with patch.dict(POLICY, max_cost=cost - 1e-6):
            self.assertEqual(len(associate(records, distance=constant_distance(.5))), 3)

    def test_frame_gap_limit_drops_old_tracks(self):
        records = [frame(f"g{k}", at(k), people=[person("p", 300, appearance=look(10))] if k in (0, 4) else [])
                   for k in range(5)]
        self.assertEqual(len(associate(records, max_frame_gap=3)), 2)
        self.assertEqual(associate(records, max_frame_gap=4)[0]["frames"], [0, 4])

    def test_crowd_invariants_and_json(self):
        records = crowd(1)
        seen = sorted((r["image_id"], p["person_id"]) for r in records for p in r["people"])
        for name, context in matchers():
            with self.subTest(matcher=name), context():
                tracks = associate(records)
                observations = [o for t in tracks for o in t["observations"]]
                self.assertEqual(sorted(observations), seen)  # every person exactly once
                for track in tracks:
                    steps = [b - a for a, b in zip(track["frames"], track["frames"][1:])]
                    self.assertTrue(all(1 <= s <= 2 for s in steps))
                    self.assertEqual(len(track["link_costs"]), len(track["frames"]) - 1)
                    self.assertTrue(all(c <= POLICY["max_cost"] for c in track["link_costs"]))
                for track in tracks:  # distinct looks: a track never mixes two walkers
                    self.assertEqual(len({pid for _, pid in track["observations"]}), 1)
                runs = 0  # a walker missing from > max_frame_gap - 1 consecutive frames starts a new track
                for pid in {pid for _, pid in seen}:
                    present = [k for k, r in enumerate(records) if any(p["person_id"] == pid for p in r["people"])]
                    runs += 1 + sum(b - a > 2 for a, b in zip(present, present[1:]))
                self.assertEqual(len(tracks), runs)
                summary = summarize_event(records, tracks)
                self.assertGreaterEqual(summary["people_unique"], summary["people_max_frame"])
                self.assertEqual(sum(summary["dir_" + d] for d in ("left", "right", "toward", "away",
                                                                    "stationary", "unclear")), len(tracks))
                self.assertEqual(summary["adults"] + summary["children"] + summary["age_unknown"], len(tracks))
                self.assertTrue(plain([tracks, summary]))
                json.dumps([tracks, summary], allow_nan=False)

    def test_malformed_people_and_ids_do_not_crash(self):
        weird = {"person_id": "w", "box": [float("nan")] * 4, "foot": [float("inf"), 1],
                 "height_px": float("nan"), "appearance": {"junk": True}}
        records = [frame("m0", at(0), people=[None, 5, {"person_id": ["list"], "box": [0, 0, 10, 50]}, weird]),
                   {**frame("m1", at(1)), "people": {"not": "a list"}},
                   frame("m2", at(2), people=[{"person_id": "w", "box": [0, 0, 0, 0], "facing": ["left"],
                                               "age": {"age": "child"}}])]
        tracks = associate(records)
        summary = summarize_event(records, tracks)
        json.dumps([tracks, summary], allow_nan=False)
        self.assertEqual(summary["people_max_frame"], 2)
        self.assertEqual(summary["people_unique"], len(tracks))
        self.assertIn(("m0", ["list"]), [o for t in tracks for o in t["observations"]])

    def test_numpy_scalars_do_not_leak_into_outputs(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        records = [frame(f"n{k}", at(k), np.int64(10 + k),
                         [{"person_id": "p", "box": [np.float64(100 + 30 * k), np.float32(200),
                                                     np.float64(140 + 30 * k), np.float64(350)],
                           "foot": [np.float64(120 + 30 * k), np.float64(350)], "height_px": np.float32(150),
                           "appearance": look(1), "facing": "left", "age": "adult"}],
                         objects={"dogs": np.int64(2), "bicycles": np.float64(1.0)})
                   for k in range(3)]
        (event,) = group_events(records)
        tracks = associate(event)
        summary = summarize_event(event, tracks)
        self.assertTrue(plain(tracks))
        self.assertTrue(plain(summary))
        self.assertEqual((summary["dogs"], summary["bicycles"]), (2, 1))

    def test_busy_event_association_is_fast(self):
        rng = random.Random(2)
        looks = [descriptor(rng) for _ in range(6)]
        records = [frame(f"b{k:03d}", at(k), people=[person(f"p{n}", 100 + 250 * n + 10 * k, appearance=looks[n])
                                                     for n in range(6)]) for k in range(60)]
        start = time.perf_counter()
        tracks = associate(records)
        self.assertLess(time.perf_counter() - start, 8.0)
        self.assertEqual(len(tracks), 6)
        self.assertFalse(any(t["ambiguous"] for t in tracks))


class AdversarialDirectionTests(unittest.TestCase):
    def test_extreme_geometry_is_finite_or_falls_back(self):
        huge_ratio = track_direction([{"foot": [0, 0], "height_px": 1e300}, {"foot": [0, 0], "height_px": 1e-300}])
        self.assertEqual((huge_ratio["direction"], huge_ratio["source"]), ("away", "motion"))
        self.assertAlmostEqual(huge_ratio["radial"], -600 * math.log(10), places=3)
        tiny = track_direction([{"foot": [0, 0], "height_px": 1e-310, "facing": "left"},
                                {"foot": [1, 0], "height_px": 1e-310}])
        self.assertEqual((tiny["direction"], tiny["source"], tiny["lateral"]), ("left", "facing", None))
        far = track_direction([{"foot": [-1e308, 0], "height_px": 100}, {"foot": [1e308, 0], "height_px": 100}])
        self.assertEqual((far["direction"], far["source"]), ("unclear", "none"))
        json.dumps([huge_ratio, tiny, far], allow_nan=False)

    def test_pairs_with_records_and_malformed_items(self):
        (event,) = group_events(walkers())
        tracks = associate(event)
        pairs = tracks[0]["observations"]
        self.assertEqual(track_direction(pairs, event), tracks[0]["direction"])
        self.assertEqual(track_direction(json.loads(json.dumps(pairs)), event), tracks[0]["direction"])
        self.assertEqual(track_direction(["junk", None, 3])["source"], "none")
        self.assertEqual(track_direction("left")["source"], "none")
        self.assertEqual(track_direction(iter([{"facing": "away"}]))["direction"], "away")

    def test_track_age_accepts_none_generators_and_junk(self):
        self.assertEqual(track_age(None), "unknown")
        self.assertEqual(track_age(o for o in [{"age": "child"}, None, "adult"]), "child")
        self.assertEqual(track_age([{"age": ["adult"]}]), "unknown")


class AdversarialSummaryTests(unittest.TestCase):
    def test_review_reasons_explain_flags(self):
        records = [frame("f0", at(0), people=[person("x", 300, 100.0, appearance={})]),
                   frame("f1", at(1), people=[person("y", 300, 170.0, appearance={})])]
        summary = summarize_event(records, associate(records, distance=constant_distance(.60)))
        self.assertTrue(summary["needs_review"])
        self.assertEqual(summary["review_reasons"], ["near_gate_candidate", "people_unique_vs_max_frame"])
        pair = [frame(f"f{k}", at(k), people=[person("p", 300 + 10 * k), person("q", 320 + 10 * k)]) for k in range(2)]
        self.assertEqual(summarize_event(pair, associate(pair, distance=constant_distance(None)))["review_reasons"],
                         ["competing_candidate", "missing_appearance"])

    def test_junk_tracks_are_ignored_and_iterables_accepted(self):
        (event,) = group_events(walkers())
        tracks = associate(event)
        self.assertEqual(summarize_event(event, tuple(tracks) + (None, "t9", 5))["people_unique"], 2)
        self.assertEqual(summarize_event(event, (t for t in tracks))["people_unique"], 2)
        self.assertEqual(summarize_event(iter(event))["people_unique"], 2)

    def test_object_edge_values(self):
        objects = [{"dogs": "3", "bicycles": float("nan")},
                   {"dogs": "٢", "bicycles": float("inf"), "strollers": "²"},
                   [1, 2], {"dogs": 10 ** 12, "motorcycles": 2.5}]
        records = [frame(f"o{k}", at(k), objects=o) for k, o in enumerate(objects)]
        summary = summarize_event(records)
        self.assertEqual(summary["dogs"], 10 ** 12)
        self.assertIsNone(summary["bicycles"])
        self.assertIsNone(summary["strollers"])
        self.assertIsNone(summary["motorcycles"])
        json.dumps(summary, allow_nan=False)


class ConservativeAssociationRegressionTests(unittest.TestCase):
    def test_distant_people_never_merge_even_with_identical_or_unavailable_looks(self):
        import numpy as np
        from trailcam.appearance import descriptor as image_descriptor
        dark = image_descriptor(np.zeros((300, 2000, 3), dtype=np.uint8), [0, 100, 50, 200], device="cpu")
        self.assertFalse(dark["valid"])
        for appearance in (dark, None, look(10)):
            for matcher, context in matchers():
                with self.subTest(appearance=appearance is None, matcher=matcher), context():
                    frames = [frame("a", at(0), people=[person("a", 25, 100, 200, appearance, "toward")]),
                              frame("b", at(1), people=[person("b", 1825, 100, 200, appearance, "away")])]
                    group = group_events(frames)[0]
                    tracks = associate(group)
                    self.assertEqual(len(tracks), 2)
                    self.assertTrue(all(t["direction"]["source"] == "facing" for t in tracks))
                    self.assertEqual(summarize_event(group, tracks)["direction_from_motion"], 0)

    def test_displacement_hard_gate_respects_the_elapsed_time_allowance(self):
        first = person("a", 100, height=100, appearance=look(10))
        elapsed = 1.0
        allowance = 100 * (1 + POLICY["displacement_seconds_factor"] * elapsed)
        self.assertTrue(link_cost(first, person("b", 100 + allowance, 100, appearance=look(10)), elapsed)[1])
        self.assertFalse(link_cost(first, person("b", 100 + allowance + 1, 100, appearance=look(10)), elapsed)[1])
        self.assertTrue(link_cost(first, person("b", 100 + allowance + 1, 100, appearance=look(10)), elapsed + 1)[1])

    def test_close_missing_appearance_links_are_flagged_and_do_not_supply_motion(self):
        frames = [frame(f"f{k}", at(k), people=[person("p", 300 + 60 * k, facing="toward")]) for k in range(3)]
        tracks = associate(frames)
        self.assertEqual(len(tracks), 1)
        self.assertIn("missing_appearance", tracks[0]["reasons"])
        self.assertTrue(tracks[0]["ambiguous"])
        self.assertEqual(tracks[0]["direction"]["source"], "facing")
        self.assertEqual(track_direction(tracks[0], frames)["source"], "facing")
        summary = summarize_event(frames, tracks)
        self.assertTrue(summary["needs_review"])
        self.assertEqual(summary["direction_from_motion"], 0)
        self.assertEqual(summary["dir_toward"], 1)

    def test_resized_frames_start_a_new_event_and_never_create_false_motion(self):
        frames = []
        for i, scale in enumerate((1., 1.5)):
            observed = person("p", 950 * scale, 200 * scale, 600 * scale, look(10), "toward")
            frames.append({**frame(str(i), at(i), people=[observed]),
                           "width": int(1920 * scale), "height": int(1080 * scale)})
        self.assertEqual([len(e) for e in group_events(frames)], [1, 1])
        tracks = associate(frames)  # callers bypassing group_events are protected too
        self.assertEqual(len(tracks), 2)
        self.assertTrue(all(t["direction"]["source"] == "facing" for t in tracks))

    def test_camera_metadata_changes_are_separate_coordinate_domains(self):
        frames = [{**frame("a", at(0), people=[person("p", 300, appearance=look(10))]), "camera_model": "A"},
                  {**frame("b", at(1), people=[person("p", 300, appearance=look(10))]), "camera_model": "B"}]
        self.assertEqual(len(group_events(frames)), 2)
        self.assertEqual(len(associate(frames)), 2)


class BackpackEventTests(unittest.TestCase):
    def test_backpacks_use_per_frame_maxima_and_preserve_unassessed(self):
        frames = [frame("a", at(0), objects={"backpacks": 1}),
                  frame("b", at(1), objects={"backpacks": 3}),
                  frame("c", at(2), objects={"backpacks": 2})]
        self.assertEqual(summarize_event(frames)["backpacks"], 3)
        self.assertEqual(summarize_event([frame("empty", at(0), objects={"backpacks": 0})])["backpacks"], 0)
        self.assertIsNone(summarize_event([frame("unassessed", at(0))])["backpacks"])


if __name__ == "__main__":
    unittest.main()
