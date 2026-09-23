"""Large-bag geometry rules (bags_v2) on synthetic boxes and keypoints only.

Fixture geometry: PERSON is 300 px tall and ``keypoints()`` puts the shoulder
line at y=160 and the hip line at y=260, so the torso T is 100 px and the
person counts as full body (300 >= 2.6 x T). With T = 100 the v2 cues read
directly off a backpack box [x1, y1, x2, y2]:
    F1 = (y2 - y1) / 100      F2 = (160 - y1) / 100      F3 = (y2 - y1) / 300
"""
import ast
import copy
import json
import math
import random
import time
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

from trailcam import bags as bags_module
from trailcam.bags import POLICY, RULE_VERSION, body_lines, classify_person_bags, count_large_bags

PERSON = [100.0, 100.0, 200.0, 400.0]   # height 300: full body with the default torso
PARTIAL = [100.0, 100.0, 200.0, 350.0]  # height 250 < 2.6 x 100: legs cut off, not full body

BACKPACK_RULES = {
    True: {"backpack_tall_and_above_shoulders_large", "backpack_full_body_height_ratio_large"},
    False: {"backpack_daypack", "backpack_daypack_height_only"},
    None: {"backpack_uncertain_mid_size", "backpack_uncertain_no_person_height", "backpack_uncertain_no_torso"},
}


def keypoints(shoulder_y=160.0, hip_y=260.0, conf=.9):
    """17 COCO triples; only shoulders (5/6) and hips (11/12) are placed."""
    points = [[150.0, 120.0, .1] for _ in range(17)]
    if shoulder_y is not None:
        points[5], points[6] = [170.0, shoulder_y, conf], [130.0, shoulder_y, conf]
    if hip_y is not None:
        points[11], points[12] = [165.0, hip_y, conf], [135.0, hip_y, conf]
    return points


def bag(label, box, confidence=.8, mask_area=None):
    return {"label": label, "box": None if box is None else list(box),
            "confidence": confidence, "mask_area": mask_area}


def backpack(y1, y2, confidence=.8, x1=120.0, x2=180.0):
    return bag("backpack", [x1, y1, x2, y2], confidence)


def verdict(person, points, detections, **kwargs):
    """(large_bag, rule of the first bag entry) for a one-person call."""
    result = classify_person_bags(person, points, detections, **kwargs)
    return result["large_bag"], result["features"]["bags"][0]["rule"]


def plain(value):
    """True when a value tree holds only JSON-native Python types."""
    if isinstance(value, dict):
        return all(type(k) is str and plain(v) for k, v in value.items())
    if isinstance(value, list):
        return all(plain(v) for v in value)
    return value is None or type(value) in (bool, int, float, str)


class PolicyTests(unittest.TestCase):
    def test_policy_is_the_bags_v2_contract(self):
        self.assertEqual(RULE_VERSION, "bags_v2")
        self.assertEqual(POLICY, {
            "keypoint_confidence": .50, "luggage_confidence": .35, "luggage_min_torso_ratio": .80,
            "backpack_confidence": .25, "large_torso_ratio": 1.40, "large_top_above_shoulder": .25,
            "large_height_ratio": .50, "full_body_per_torso": 2.6, "daypack_torso_ratio": 1.15,
            "daypack_top_above_shoulder": .15, "daypack_height_ratio": .30,
            "min_torso_per_person_height": .15, "duplicate_iou": .50, "version": "bags_v2"})

    def test_daypack_and_large_thresholds_leave_an_uncertain_band(self):
        # Every daypack cut sits strictly below its large counterpart, so mid-size
        # packs abstain instead of flipping between the two verdicts.
        self.assertLess(POLICY["daypack_torso_ratio"], POLICY["large_torso_ratio"])
        self.assertLess(POLICY["daypack_top_above_shoulder"], POLICY["large_top_above_shoulder"])
        self.assertLess(POLICY["daypack_height_ratio"], POLICY["large_height_ratio"])
        self.assertEqual(classify_person_bags(PERSON, keypoints(), [])["features"]["rule_version"], "bags_v2")


class TwoCueRuleTests(unittest.TestCase):
    """F1 (bag height / T) >= 1.40 together with F2 (top above shoulders / T) >= .25."""

    def test_tall_pack_rising_above_shoulders_is_large(self):
        result = classify_person_bags(PERSON, keypoints(), [backpack(135, 275)])
        self.assertIs(result["large_bag"], True)
        self.assertIn("backpack_tall_and_above_shoulders_large", result["reasons"])
        entry = result["features"]["bags"][0]
        self.assertEqual({k: entry[k] for k in ("bag_height_px", "torso_ratio", "top_above_shoulder_px",
                                                "top_above_shoulder_per_torso", "height_ratio", "full_body")},
                         {"bag_height_px": 140.0, "torso_ratio": 1.4, "top_above_shoulder_px": 25.0,
                          "top_above_shoulder_per_torso": .25, "height_ratio": .466667, "full_body": True})
        self.assertEqual((entry["status"], entry["verdict"]), ("large", True))
        self.assertEqual(result["features"]["large_bag_count"], 1)

    def test_both_cue_boundaries_are_inclusive(self):
        # PARTIAL keeps the full-body F3 rule out of the way (F3 would be >= .5 here).
        cases = {(135.0, 275.0): (True, "backpack_tall_and_above_shoulders_large"),    # F1 1.40, F2 .25
                 (135.0, 274.9): (None, "backpack_uncertain_mid_size"),                # F1 1.399
                 (135.1, 275.1): (None, "backpack_uncertain_mid_size")}                # F2 .249
        for (y1, y2), expected in cases.items():
            with self.subTest(y1=y1, y2=y2):
                self.assertEqual(verdict(PARTIAL, keypoints(), [backpack(y1, y2)]), expected)

    def test_a_single_cue_is_not_enough(self):
        # v1 called each of these large on one cue; v2 needs two agreeing cues.
        rising_only = verdict(PERSON, keypoints(), [backpack(100, 200)])      # F2 .60, F1 1.00
        tall_only = verdict(PERSON, keypoints(), [backpack(160, 300)])        # F1 1.40, F2 0, F3 .47
        self.assertEqual(rising_only, (None, "backpack_uncertain_mid_size"))
        self.assertEqual(tall_only, (None, "backpack_uncertain_mid_size"))
        # v1's frame-pack example: 70 px pack 15 px above the shoulders is a daypack now.
        self.assertEqual(verdict(PERSON, keypoints(), [backpack(145, 215)]), (False, "backpack_daypack"))

    def test_two_cue_rule_takes_precedence_over_full_body_rule(self):
        result = classify_person_bags(PERSON, keypoints(), [backpack(100, 260)])  # F1 1.6, F2 .6, F3 .53
        self.assertIs(result["large_bag"], True)
        self.assertEqual(result["features"]["bags"][0]["rule"], "backpack_tall_and_above_shoulders_large")
        self.assertNotIn("backpack_full_body_height_ratio_large", result["reasons"])


class FullBodyRuleTests(unittest.TestCase):
    """F3 (bag height / person height) >= .50 alone, only with person height >= 2.6 x T."""

    def test_full_body_height_ratio_rule_and_boundary(self):
        result = classify_person_bags(PERSON, keypoints(), [backpack(170, 320)])  # F3 .50, F2 -.10
        self.assertIs(result["large_bag"], True)
        self.assertIn("backpack_full_body_height_ratio_large", result["reasons"])
        entry = result["features"]["bags"][0]
        self.assertEqual((entry["height_ratio"], entry["full_body"]), (.5, True))
        self.assertEqual(verdict(PERSON, keypoints(), [backpack(170, 319.9)]),   # F3 .4997
                         (None, "backpack_uncertain_mid_size"))

    def test_height_ratio_alone_needs_the_full_body(self):
        # 150 px pack on a 250 px person (F3 .60): legs are cut off, so the person
        # box does not measure stature and F3 is not trusted.
        result = classify_person_bags(PARTIAL, keypoints(), [backpack(170, 320)])
        self.assertIsNone(result["large_bag"])
        entry = result["features"]["bags"][0]
        self.assertEqual((entry["height_ratio"], entry["full_body"], entry["rule"]),
                         (.6, False, "backpack_uncertain_mid_size"))

    def test_full_body_gate_boundary_is_inclusive(self):
        exact = [100.0, 100.0, 200.0, 360.0]    # 260 px = 2.6 x 100
        short = [100.0, 100.1, 200.0, 360.0]    # 259.9 px
        self.assertEqual(verdict(exact, keypoints(), [backpack(170, 300)]),    # F3 = 130 / 260 = .50
                         (True, "backpack_full_body_height_ratio_large"))
        result = classify_person_bags(short, keypoints(), [backpack(170, 300)])  # F3 .5002
        self.assertIsNone(result["large_bag"])
        self.assertIs(result["features"]["bags"][0]["full_body"], False)

    def test_full_body_gate_is_compared_at_reported_precision(self):
        # v2: person height / torso is rounded to 6 decimals before the 2.6 check:
        # 260 px over a 100.0000001 px torso (2.599999997) is full body; 100.01 px is not.
        person = [100.0, 100.0, 200.0, 360.0]
        for hip_y, full in ((260 + 1e-7, True), (260.01, False)):
            with self.subTest(hip_y=hip_y):
                result = classify_person_bags(person, keypoints(hip_y=hip_y), [backpack(160, 290)])
                entry = result["features"]["bags"][0]
                self.assertIs(entry["full_body"], full)
                self.assertEqual(entry["rule"], "backpack_full_body_height_ratio_large" if full
                                 else "backpack_uncertain_mid_size")

    def test_full_body_requires_a_measured_torso(self):
        for points in (None, keypoints(hip_y=None), keypoints(shoulder_y=None)):
            with self.subTest(points=None if points is None else "partial"):
                result = classify_person_bags(PERSON, points, [backpack(100, 400)])   # F3 1.0
                self.assertIsNone(result["large_bag"])
                self.assertIs(result["features"]["bags"][0]["full_body"], False)
                self.assertEqual(result["features"]["bags"][0]["rule"], "backpack_uncertain_no_torso")


class DaypackRuleTests(unittest.TestCase):
    def test_daypack_with_torso_boundaries(self):
        cases = {(160.0, 275.0): (False, "backpack_daypack"),              # F1 1.15, F2 0
                 (160.0, 275.1): (None, "backpack_uncertain_mid_size"),    # F1 1.151
                 (145.0, 245.0): (False, "backpack_daypack"),              # F1 1.00, F2 .15
                 (144.9, 245.0): (None, "backpack_uncertain_mid_size"),    # F2 .151
                 (300.0, 360.0): (False, "backpack_daypack")}              # entirely below the hips
        for (y1, y2), expected in cases.items():
            with self.subTest(y1=y1, y2=y2):
                self.assertEqual(verdict(PERSON, keypoints(), [backpack(y1, y2)]), expected)

    def test_daypack_counts_in_features(self):
        result = classify_person_bags(PERSON, keypoints(), [backpack(165, 225)])
        self.assertIs(result["large_bag"], False)
        self.assertIn("backpack_daypack", result["reasons"])
        self.assertEqual([result["features"][k] for k in ("large_bag_count", "uncertain_bag_count", "daypack_count")],
                         [0, 0, 1])

    def test_mid_size_pack_is_uncertain(self):
        result = classify_person_bags(PERSON, keypoints(), [backpack(160, 290)])  # F1 1.30
        self.assertIsNone(result["large_bag"])
        self.assertIn("backpack_uncertain_mid_size", result["reasons"])
        self.assertEqual(result["features"]["uncertain_bag_count"], 1)

    def test_height_ratio_rules_without_torso(self):
        cases = {90.0: (False, "backpack_daypack_height_only"),   # F3 .30
                 90.1: (None, "backpack_uncertain_no_torso"),     # F3 .3003
                 200.0: (None, "backpack_uncertain_no_torso"),    # F3 .67: never large without a torso
                 300.0: (None, "backpack_uncertain_no_torso")}
        for height, expected in cases.items():
            with self.subTest(height=height):
                self.assertEqual(verdict(PERSON, None, [backpack(100, 100 + height)]), expected)

    def test_missing_keypoints_use_height_ratio_rules_only(self):
        self.assertIs(classify_person_bags(PERSON, keypoints(), [backpack(135, 275)])["large_bag"], True)
        for missing in (None, [], keypoints(None, None), keypoints(conf=.49)):
            with self.subTest(keypoints=missing):
                result = classify_person_bags(PERSON, missing, [backpack(135, 275)])  # F3 .47
                self.assertIsNone(result["large_bag"])
                self.assertIn("height_ratio_rules_only", result["reasons"])
                self.assertIsNone(result["features"]["torso_px"])
                self.assertEqual(result["features"]["bags"][0]["rule"], "backpack_uncertain_no_torso")

    def test_shoulders_without_hips_use_height_ratio_rules(self):
        points = keypoints(hip_y=None)
        self.assertIsNone(body_lines(points)["torso_px"])
        # 20 px above the shoulder line is reported, but with no torso to scale it
        # only the height ratio (.20) decides.
        result = classify_person_bags(PERSON, points, [backpack(140, 200)])
        self.assertIs(result["large_bag"], False)
        self.assertIn("hips_not_visible", result["reasons"])
        entry = result["features"]["bags"][0]
        self.assertEqual((entry["rule"], entry["top_above_shoulder_px"], entry["top_above_shoulder_per_torso"],
                          entry["torso_ratio"]), ("backpack_daypack_height_only", 20.0, None, None))
        self.assertEqual(verdict(PERSON, points, [backpack(135, 275)]), (None, "backpack_uncertain_no_torso"))

    def test_hips_without_shoulders_use_height_ratio_rules(self):
        result = classify_person_bags(PERSON, keypoints(shoulder_y=None), [backpack(145, 215)])
        self.assertIs(result["large_bag"], False)
        self.assertIn("shoulders_not_visible", result["reasons"])
        self.assertIn("height_ratio_rules_only", result["reasons"])
        self.assertIsNone(result["features"]["bags"][0]["top_above_shoulder_px"])


class BodyLineTests(unittest.TestCase):
    def test_keypoint_confidence_threshold_is_inclusive(self):
        lines = body_lines(keypoints(conf=.5))
        self.assertEqual((lines["shoulder_y"], lines["hip_y"], lines["torso_px"]), (160.0, 260.0, 100.0))
        self.assertEqual(lines["used"], ["left_shoulder", "right_shoulder", "left_hip", "right_hip"])
        self.assertIsNone(body_lines(keypoints(conf=.499))["shoulder_y"])
        # Six-decimal comparison: .4999996 reads as .5, .499999 does not.
        self.assertEqual(body_lines(keypoints(conf=.4999996))["torso_px"], 100.0)
        self.assertIsNone(body_lines(keypoints(conf=.499999))["torso_px"])

    def test_single_visible_shoulder_and_hip_define_lines(self):
        points = keypoints()
        points[6][2], points[11][2] = .1, .1
        points[5][1], points[12][1] = 158.0, 262.0
        lines = body_lines(points)
        self.assertEqual((lines["shoulder_y"], lines["hip_y"], lines["torso_px"]), (158.0, 262.0, 104.0))

    def test_inverted_torso_discards_keypoints(self):
        result = classify_person_bags(PERSON, keypoints(shoulder_y=260, hip_y=160), [backpack(145, 215)])
        self.assertIs(result["large_bag"], False)
        self.assertIn("torso_inverted_keypoints_discarded", result["reasons"])
        self.assertEqual(result["features"]["keypoints_used"], [])
        self.assertEqual(result["features"]["bags"][0]["rule"], "backpack_daypack_height_only")

    def test_short_torso_is_not_used_for_rules(self):
        # Bent-over hiker: a 30 px vertical torso (.10 of height) would make this
        # 50 px pack read F1 1.67 / F2 1.67 (large); only the height ratio is used.
        result = classify_person_bags(PERSON, keypoints(shoulder_y=200, hip_y=230), [backpack(150, 200)])
        self.assertIs(result["large_bag"], False)
        self.assertIn("torso_too_short_for_rules", result["reasons"])
        self.assertIsNone(result["features"]["torso_px"])
        self.assertEqual(result["features"]["bags"][0]["rule"], "backpack_daypack_height_only")
        self.assertIs(result["features"]["bags"][0]["full_body"], False)

    def test_short_torso_boundary_is_inclusive(self):
        kept = classify_person_bags(PERSON, keypoints(shoulder_y=160, hip_y=205), [backpack(100, 170)])
        self.assertEqual(kept["features"]["torso_px"], 45.0)  # 45 / 300 = .15 exactly
        self.assertIs(kept["large_bag"], True)                # F1 1.56, F2 1.33
        dropped = classify_person_bags(PERSON, keypoints(shoulder_y=160, hip_y=204.9), [backpack(100, 170)])
        self.assertIsNone(dropped["features"]["torso_px"])
        self.assertIs(dropped["large_bag"], False)            # F3 .23
        self.assertIn("torso_too_short_for_rules", dropped["reasons"])

    def test_equal_hip_and_shoulder_lines_are_discarded(self):
        lines = body_lines(keypoints(shoulder_y=200, hip_y=200))
        self.assertEqual((lines["shoulder_y"], lines["hip_y"], lines["used"]), (None, None, []))
        self.assertIn("torso_inverted_keypoints_discarded", lines["reasons"])

    def test_invalid_keypoint_shapes_are_treated_as_missing(self):
        for broken in (keypoints()[:16], [[1.0, 2.0]] * 17, "not keypoints", 5):
            with self.subTest(keypoints=broken):
                result = classify_person_bags(PERSON, broken, [backpack(135, 275)])
                self.assertIsNone(result["large_bag"])
                self.assertIn("height_ratio_rules_only", result["reasons"])

    def test_nonfinite_keypoints_are_unusable(self):
        points = keypoints()
        points[5][1] = points[6][1] = math.nan
        self.assertIsNone(body_lines(points)["shoulder_y"])
        self.assertEqual(body_lines(points)["hip_y"], 260.0)
        points[11][2] = points[12][2] = math.inf  # an infinite confidence is not trusted
        self.assertIsNone(body_lines(points)["hip_y"])


class LuggageTests(unittest.TestCase):
    def test_confident_luggage_without_torso_is_large_regardless_of_size(self):
        for points in (None, keypoints(shoulder_y=200, hip_y=230)):  # missing / too-short torso
            with self.subTest(points=points is not None):
                result = classify_person_bags(PERSON, points, [bag("suitcase", [210, 330, 240, 350], .6)])
                self.assertIs(result["large_bag"], True)
                self.assertIn("suitcase_confident_large", result["reasons"])

    def test_luggage_long_side_must_reach_torso_ratio(self):
        cases = {(210, 320, 250, 400): (True, "suitcase_confident_large"),      # long side 80 = .8 T
                 (210, 320.5, 250, 400): (None, "suitcase_small_for_luggage"),  # 79.5
                 (100, 380, 180, 400): (True, "suitcase_confident_large"),      # width is the long side
                 (100, 380, 179.5, 400): (None, "suitcase_small_for_luggage")}
        for box, expected in cases.items():
            with self.subTest(box=box):
                self.assertEqual(verdict(PERSON, keypoints(), [bag("suitcase", box)]), expected)
        small = classify_person_bags(PERSON, keypoints(), [bag("duffel bag", [210, 330, 250, 400])])
        self.assertIsNone(small["large_bag"])
        self.assertEqual(small["features"]["bags"][0]["status"], "uncertain")
        self.assertEqual(small["features"]["uncertain_bag_count"], 1)
        self.assertIn("duffel_bag_small_for_luggage", small["reasons"])

    def test_size_check_uses_torso_even_without_person_height(self):
        self.assertEqual(verdict(None, keypoints(), [bag("suitcase", [0, 0, 10, 10])]),
                         (None, "suitcase_small_for_luggage"))

    def test_luggage_confidence_boundary_is_inclusive(self):
        self.assertIs(classify_person_bags(PERSON, None, [bag("duffel bag", [0, 0, 10, 10], .35)])["large_bag"], True)
        self.assertEqual(verdict(PERSON, keypoints(), [bag("suitcase", [210, 320, 250, 400], .35)]),
                         (True, "suitcase_confident_large"))

    def test_low_confidence_luggage_is_ignored_before_size_check(self):
        low = bag("duffel bag", [120, 250, 190, 300], .34)
        result = classify_person_bags(PERSON, keypoints(), [low])
        self.assertIsNone(result["large_bag"])
        self.assertIn("duffel_bag_low_confidence_ignored", result["reasons"])
        self.assertNotIn("duffel_bag_small_for_luggage", result["reasons"])
        self.assertIn("no_usable_bag_person_not_assessable", result["reasons"])
        self.assertIs(classify_person_bags(PERSON, keypoints(), [low], assessable=True)["large_bag"], False)
        # Ignored duffel does not override a daypack.
        mixed = classify_person_bags(PERSON, keypoints(), [low, backpack(165, 225)])
        self.assertIs(mixed["large_bag"], False)
        self.assertEqual(mixed["features"]["bags"][0]["status"], "ignored")

    def test_label_normalisation_and_xyxy_key(self):
        detection = {"label": " Duffel_Bag ", "xyxy": [0, 0, 20, 20], "confidence": .9}
        self.assertIs(classify_person_bags(PERSON, None, [detection])["large_bag"], True)

    def test_luggage_entries_report_size_per_torso(self):
        # v2: the luggage size cue is reported (6 decimals) like the backpack cues.
        entry = classify_person_bags(PERSON, keypoints(), [bag("suitcase", [210, 320, 250, 400])])["features"]["bags"][0]
        self.assertEqual((entry["luggage_size_per_torso"], entry["status"]), (.8, "large"))
        small = classify_person_bags(PERSON, keypoints(), [bag("suitcase", [0, 0, 10, 30])])["features"]["bags"][0]
        self.assertEqual((small["luggage_size_per_torso"], small["status"]), (.3, "uncertain"))
        untorsoed = classify_person_bags(PERSON, None, [bag("duffel bag", [0, 0, 10, 30])])["features"]["bags"][0]
        self.assertIsNone(untorsoed["luggage_size_per_torso"])  # no torso: size cannot be judged
        backpack_entry = classify_person_bags(PERSON, keypoints(), [backpack(165, 225)])["features"]["bags"][0]
        self.assertNotIn("luggage_size_per_torso", backpack_entry)

    def test_luggage_size_is_compared_at_reported_precision(self):
        # v2: with a torso of 100.0000001 px an 80 px suitcase measures 0.7999999992,
        # which reports as 0.8 and must decide as 0.8 (inclusive), not as "small".
        for hip_y, expected in ((260 + 1e-7, True), (260.01, None)):
            with self.subTest(hip_y=hip_y):
                result = classify_person_bags(PERSON, keypoints(hip_y=hip_y), [bag("suitcase", [210, 320, 250, 400])])
                self.assertIs(result["large_bag"], expected)
                entry = result["features"]["bags"][0]
                self.assertEqual(entry["luggage_size_per_torso"], .8 if expected else .79992)
                self.assertEqual(entry["luggage_size_per_torso"] >= POLICY["luggage_min_torso_ratio"], bool(expected))


class InputEdgeCaseTests(unittest.TestCase):
    def test_no_bag_depends_on_assessability(self):
        self.assertIsNone(classify_person_bags(PERSON, keypoints(), [])["large_bag"])
        self.assertIsNone(classify_person_bags(PERSON, keypoints(), None)["large_bag"])
        self.assertIs(classify_person_bags(PERSON, keypoints(), [], assessable=True)["large_bag"], False)
        self.assertIsNone(classify_person_bags(PERSON, keypoints(), [], assessable=None)["large_bag"])
        self.assertIs(classify_person_bags(PERSON, keypoints(), [], assessable=np.bool_(True))["large_bag"], False)
        result = classify_person_bags(PERSON, keypoints(), [], True)
        self.assertIn("no_usable_bag_person_assessable", result["reasons"])
        self.assertEqual(result["features"]["large_bag_count"], 0)

    def test_ignored_detections(self):
        ignored = [bag("handbag", [120, 200, 150, 260], .9),
                   backpack(165, 225, confidence=.2),
                   backpack(165, 225, confidence=None),
                   bag("backpack", [120, 200, 120, 260]),          # zero width
                   bag("suitcase", [120, 260, 150, 200]),          # inverted
                   bag("suitcase", [120, math.nan, 150, 260]),
                   bag("suitcase", [1, 2, 3]),
                   bag("suitcase", None),
                   "not a detection", None]
        result = classify_person_bags(PERSON, keypoints(), ignored, assessable=True)
        self.assertIs(result["large_bag"], False)
        rules = [entry["rule"] for entry in result["features"]["bags"]]
        self.assertEqual(rules, ["unsupported_label_ignored", "backpack_low_confidence_ignored",
                                 "bag_confidence_missing_ignored"] + ["bag_box_invalid_ignored"] * 5)
        self.assertTrue(all(entry["status"] == "ignored" for entry in result["features"]["bags"]))

    def test_degenerate_person_box_keeps_torso_rules(self):
        for box in ([100, 100, 200, 100], None, [100, 100, 200], [0, 0, math.inf, 10]):
            with self.subTest(box=box):
                self.assertEqual(verdict(box, keypoints(), [backpack(135, 275)]),
                                 (True, "backpack_tall_and_above_shoulders_large"))
                small = classify_person_bags(box, keypoints(), [backpack(165, 225)])
                self.assertIs(small["large_bag"], False)
                self.assertIn("person_box_invalid", small["reasons"])
                self.assertEqual(small["features"]["bags"][0]["rule"], "backpack_daypack")
                self.assertIsNone(small["features"]["person_height_px"])
                # No person height: never full body, so F3 cannot decide.
                self.assertEqual(verdict(box, keypoints(), [backpack(170, 320)]), (None, "backpack_uncertain_mid_size"))
                self.assertEqual(verdict(box, None, [backpack(165, 225)]),
                                 (None, "backpack_uncertain_no_person_height"))

    def test_multiple_bags_combine_large_over_uncertain_over_daypack(self):
        daypack, mid = backpack(165, 225), backpack(160, 290, x1=60, x2=100)
        large = bag("suitcase", [210, 310, 250, 400])  # long side 90 >= .8 T
        self.assertIsNone(classify_person_bags(PERSON, keypoints(), [daypack, mid])["large_bag"])
        result = classify_person_bags(PERSON, keypoints(), [daypack, mid, large])
        self.assertIs(result["large_bag"], True)
        self.assertEqual([result["features"][k] for k in ("large_bag_count", "uncertain_bag_count", "daypack_count")],
                         [1, 1, 1])

    def test_duplicate_boxes_are_one_bag(self):
        twins = [bag("suitcase", [210, 330, 250, 400], .5), bag("suitcase", [211, 331, 250, 400], .7)]
        result = classify_person_bags(PERSON, None, twins)
        self.assertEqual(result["features"]["large_bag_count"], 1)
        self.assertEqual(result["features"]["bags"][0]["duplicate_of"], 1)  # higher confidence kept
        self.assertIn("duplicate_bag_box_merged", result["reasons"])
        # Same physical bag read as a daypack by one label, confident duffel by another.
        pair = [backpack(165, 225), bag("duffel bag", [121, 166, 180, 225], .5)]
        mixed = classify_person_bags(PERSON, None, pair)
        self.assertIs(mixed["large_bag"], True)
        self.assertEqual((mixed["features"]["large_bag_count"], mixed["features"]["daypack_count"]), (1, 0))
        # With a torso the 59 px duffel is small for luggage: the uncertain reading
        # still outranks the daypack reading of the same bag.
        sized = classify_person_bags(PERSON, keypoints(), pair)
        self.assertIsNone(sized["large_bag"])
        self.assertEqual([sized["features"][k] for k in ("large_bag_count", "uncertain_bag_count", "daypack_count")],
                         [0, 1, 0])
        self.assertEqual(sized["features"]["bags"][0]["duplicate_of"], 1)
        separate = classify_person_bags(PERSON, None, [bag("suitcase", [210, 330, 250, 400]), bag("suitcase", [260, 330, 300, 400])])
        self.assertEqual(separate["features"]["large_bag_count"], 2)

    def test_numpy_inputs_give_json_native_deterministic_output(self):
        points = np.asarray(keypoints(), dtype=np.float32)
        bags = [{"label": np.str_("backpack"), "box": np.asarray([120, 135, 180, 275], dtype=np.float32),
                 "confidence": np.float32(.8), "mask_area": np.int64(2100)}]
        box = np.asarray(PERSON, dtype=np.float64)
        first = classify_person_bags(box, points, bags, assessable=np.bool_(True))
        self.assertIs(first["large_bag"], True)
        self.assertTrue(plain(first), first)
        self.assertEqual(first["features"]["bags"][0]["mask_area"], 2100)
        self.assertIs(first["features"]["bags"][0]["full_body"], True)
        self.assertIs(first["features"]["assessable"], True)
        self.assertEqual(json.loads(json.dumps(first)), first)
        self.assertEqual(classify_person_bags(box, points, bags, assessable=np.bool_(True)), first)

    def test_inputs_are_not_mutated(self):
        bags, points = [backpack(135, 275), bag("suitcase", [0, 0, 5, 5])], keypoints()
        before = copy.deepcopy((bags, points))
        classify_person_bags(PERSON, points, bags)
        self.assertEqual((bags, points), before)

    def test_person_features_report_body_lines(self):
        features = classify_person_bags(PERSON, keypoints(), [backpack(135, 275)])["features"]
        self.assertEqual({k: features[k] for k in ("rule_version", "person_height_px", "shoulder_y", "hip_y", "torso_px")},
                         {"rule_version": "bags_v2", "person_height_px": 300.0, "shoulder_y": 160.0,
                          "hip_y": 260.0, "torso_px": 100.0})


class CountLargeBagsTests(unittest.TestCase):
    def test_aggregates_people_and_loose_luggage(self):
        people = [classify_person_bags(PERSON, keypoints(), [backpack(135, 275)]),       # large pack
                  classify_person_bags(PERSON, keypoints(), [backpack(165, 225)]),       # daypack
                  classify_person_bags(PERSON, keypoints(), [backpack(160, 290)]),       # uncertain
                  classify_person_bags(PERSON, None, []),                               # unassessable, no bag
                  classify_person_bags(PERSON, None, [bag("suitcase", [210, 330, 250, 400]),
                                                     bag("duffel bag", [40, 330, 90, 400])])]  # two large
        loose = [bag("suitcase", [500, 300, 560, 400], .5),
                 bag("suitcase", [502, 301, 560, 400], .4),      # same suitcase, second expert
                 bag("duffel bag", [700, 350, 780, 400], .35),   # boundary confidence counts
                 bag("suitcase", [900, 300, 960, 400], .2),      # low confidence ignored
                 bag("backpack", [1000, 300, 1040, 360], .9),    # loose backpack not luggage
                 bag("suitcase", [1100, 300, 1100, 400], .9)]    # degenerate box
        self.assertEqual(count_large_bags(people, loose), {"large_bags": 1 + 2 + 2, "large_bags_uncertain": 1})

    def test_small_carried_luggage_counts_as_uncertain(self):
        carried = classify_person_bags(PERSON, keypoints(), [bag("suitcase", [210, 330, 250, 400])])  # 70 < 80
        self.assertEqual(count_large_bags([carried], []), {"large_bags": 0, "large_bags_uncertain": 1})

    def test_loose_luggage_counts_each_item_once(self):
        loose = [bag("suitcase", [0, 0, 50, 80], .9), bag("duffel bag", [2, 1, 50, 80], .6),
                 bag("suitcase", [100, 0, 150, 80], .5)]
        self.assertEqual(count_large_bags([], loose), {"large_bags": 2, "large_bags_uncertain": 0})
        self.assertEqual(count_large_bags([], list(reversed(loose))), {"large_bags": 2, "large_bags_uncertain": 0})

    def test_empty_and_none_inputs(self):
        for people, loose in (([], []), (None, None), ([None, "x"], [None, "x", {}])):
            with self.subTest(people=people, loose=loose):
                self.assertEqual(count_large_bags(people, loose), {"large_bags": 0, "large_bags_uncertain": 0})

    def test_bare_results_without_features(self):
        # A bare None may be a bagless, unassessable person: it is not an uncertain bag
        # (the same person keeps the same count whether or not features were stored).
        people = [{"large_bag": True}, {"large_bag": None}, {"large_bag": False},
                  {"large_bag": True, "features": {"large_bag_count": 0}},        # inconsistent: still one
                  {"large_bag": None, "features": {"uncertain_bag_count": 0}},     # unassessable, no bag
                  {"large_bag": None, "features": {"uncertain_bag_count": 2}},     # two undecided bags
                  {"large_bag": True, "features": {"large_bag_count": True}}]      # bool is not a count
        self.assertEqual(count_large_bags(people, []), {"large_bags": 3, "large_bags_uncertain": 2})
        stripped = classify_person_bags(PERSON, None, [])
        self.assertEqual(count_large_bags([stripped], []),
                         count_large_bags([{"large_bag": stripped["large_bag"]}], []))

    def test_output_types_are_plain_ints(self):
        loose = [{"label": "suitcase", "box": np.asarray([0, 0, 10, 10], np.float32), "confidence": np.float32(.9)}]
        counts = count_large_bags([classify_person_bags(PERSON, keypoints(), [backpack(100, 300)])], loose)
        self.assertEqual(counts, {"large_bags": 2, "large_bags_uncertain": 0})
        self.assertTrue(all(type(v) is int for v in counts.values()))


def strict_json(testcase, value):
    """Serialises without NaN/Infinity and round-trips unchanged."""
    testcase.assertEqual(json.loads(json.dumps(value, allow_nan=False)), value)


class AdversarialInputTests(unittest.TestCase):
    def test_float32_confidences_at_thresholds_are_inclusive(self):
        # float32(.35) == .3499999940: detector outputs sitting on a threshold still pass.
        suitcase = bag("suitcase", [0, 0, 10, 10], np.float32(.35))
        self.assertIs(classify_person_bags(PERSON, None, [suitcase])["large_bag"], True)
        self.assertEqual(count_large_bags([], [suitcase])["large_bags"], 1)
        weak = classify_person_bags(PERSON, keypoints(), [backpack(165, 225, confidence=np.float32(.25))])
        self.assertEqual(weak["features"]["bags"][0]["status"], "daypack")
        points = np.asarray(keypoints(conf=.5), dtype=np.float32)
        self.assertEqual(body_lines(points)["torso_px"], 100.0)
        # Six-decimal precision: .3499996 reports (and decides) as .35, .349999 does not.
        self.assertIs(classify_person_bags(PERSON, None, [bag("suitcase", [0, 0, 5, 5], .3499996)])["large_bag"], True)
        self.assertIsNone(classify_person_bags(PERSON, None, [bag("suitcase", [0, 0, 5, 5], .349999)])["large_bag"])

    def test_ratio_cues_decide_at_reported_precision(self):
        # A box edge within float noise of a ratio threshold decides exactly as the
        # reported six-decimal cue reads (F1 1.4 inclusive), never one ulp away.
        for y2 in (275 - 1e-9, 274.9999994):
            with self.subTest(y2=y2):
                result = classify_person_bags(PARTIAL, keypoints(), [backpack(135, y2)])
                self.assertEqual(result["features"]["bags"][0]["torso_ratio"], 1.4)
                self.assertIs(result["large_bag"], True)
        result = classify_person_bags(PARTIAL, keypoints(), [backpack(135, 274.9999)])
        self.assertEqual(result["features"]["bags"][0]["torso_ratio"], 1.399999)
        self.assertIsNone(result["large_bag"])


    def test_extreme_numbers_never_crash_and_stay_strict_json(self):
        huge = 10 ** 400  # int too large for float
        points = keypoints()
        points[5] = points[6] = [0.0, -1e308, .9]
        points[11] = points[12] = [0.0, 1e308, .9]   # torso overflows to inf
        cases = [
            (PERSON, None, [bag("suitcase", [0, 0, 10, 10], huge)]),
            (PERSON, None, [bag("suitcase", [0, 0, huge, 10])]),
            (PERSON, None, [bag("suitcase", [0, 0, 10, 10], mask_area=huge)]),
            ([-1e308, -1e308, 1e308, 1e308], keypoints(), [backpack(165, 225)]),  # size overflows
            ([0, 0, 1, 5e-324], None, [backpack(0, 1e10)]),                        # sub-precision height
            ([0, 0, 1e-300, 1e-300], None, [backpack(0, 1e300)]),
            ([0, 0, 1e300, 1e300], keypoints(), [backpack(0, 1e300), bag("suitcase", [0, 0, 1e300, 1e300])]),
            (PERSON, keypoints(1e-300, 2e-300), [backpack(0, 1e300)]),             # vanishing torso
            (PERSON, points, [backpack(145, 215)]),
            ([1e300, 1e300, 2e300, 3e300], None, [bag("backpack", [1e300, 1e300, 2e300, 3e300])]),
        ]
        for person, pts, detections in cases:
            with self.subTest(person=person, bags=detections):
                strict_json(self, classify_person_bags(person, pts, detections))
        lines = body_lines(points)
        self.assertIn("torso_nonfinite_keypoints_discarded", lines["reasons"])
        strict_json(self, lines)
        self.assertEqual(classify_person_bags(PERSON, None, [bag("suitcase", [0, 0, 10, 10], huge)])
                         ["features"]["bags"][0]["rule"], "bag_confidence_missing_ignored")
        self.assertEqual(classify_person_bags([0, 0, 1, 5e-324], None, [])["reasons"][0], "person_box_invalid")
        strict_json(self, count_large_bags([{"large_bag": True, "features": {"large_bag_count": huge}}],
                                           [bag("suitcase", [0, 0, 1e308, 1e308], huge)]))

    def test_malformed_keypoint_containers_are_invalid_not_fatal(self):
        names = ["k%d" % i for i in range(17)]
        broken = {"row dicts": [{"x": 1, "y": 2, "c": .9}] * 17,
                  "name keyed": {name: [150.0, 160.0, .9] for name in names},
                  "sets": [{1.0, 2.0, 3.0}] * 17,
                  "string rows": ["abc"] * 17,
                  "None rows": [None] * 17,
                  "scalar rows": np.zeros(17),
                  "batched": np.zeros((1, 17, 3)),
                  "bytes": b"x" * 17}
        for name, points in broken.items():
            with self.subTest(name):
                result = classify_person_bags(PERSON, points, [backpack(135, 275)])
                self.assertIsNone(result["large_bag"])
                self.assertIn("keypoints_invalid_shape", result["reasons"])
                strict_json(self, result)
        generated = classify_person_bags(PERSON, (row for row in keypoints()), [backpack(135, 275)])
        self.assertIs(generated["large_bag"], True)  # any iterable of 17 triples works
        extra = [row + [1.0] for row in keypoints()]  # [x, y, conf, visibility]
        self.assertEqual(body_lines(extra)["torso_px"], 100.0)
        none_values = keypoints()
        none_values[5] = [None, None, None]
        self.assertEqual(body_lines(none_values)["used"], ["right_shoulder", "left_hip", "right_hip"])

    def test_multi_dimensional_and_non_numeric_values_are_rejected_quietly(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no numpy "ndim > 0 to scalar" deprecation path
            column = {"label": "suitcase", "box": np.array([[0], [0], [10], [10]]), "confidence": .9}
            self.assertEqual(classify_person_bags(PERSON, None, [column])["features"]["bags"][0]["rule"],
                             "bag_box_invalid_ignored")
            nested_conf = {"label": "suitcase", "box": [0, 0, 10, 10], "confidence": np.array([.9])}
            self.assertEqual(classify_person_bags(PERSON, None, [nested_conf])["features"]["bags"][0]["rule"],
                             "bag_confidence_missing_ignored")
        for box in ("1234", b"1234", {"x1": 0, "y1": 0, "x2": 1, "y2": 1}, 7, [1, 2, 3, 4, 5]):
            with self.subTest(box=box):
                result = classify_person_bags(PERSON, None, [bag("suitcase", [0, 0, 1, 1]) | {"box": box}])
                self.assertEqual(result["features"]["bags"][0]["rule"], "bag_box_invalid_ignored")
        for conf in (True, np.bool_(True), "high", [.9], math.nan, -math.inf):
            with self.subTest(conf=conf):
                result = classify_person_bags(PERSON, None, [bag("suitcase", [0, 0, 10, 10], conf)])
                self.assertIsNone(result["large_bag"])
                self.assertEqual(result["features"]["bags"][0]["rule"], "bag_confidence_missing_ignored")
        self.assertIs(classify_person_bags(PERSON, None, [bag("suitcase", [0, 0, 10, 10], "0.9")])["large_bag"], True)

    def test_box_key_falls_back_to_xyxy_when_box_is_none(self):
        detection = {"label": "suitcase", "box": None, "xyxy": [0, 0, 20, 20], "confidence": .9}
        self.assertIs(classify_person_bags(PERSON, None, [detection])["large_bag"], True)
        self.assertEqual(count_large_bags([], [detection])["large_bags"], 1)

    def test_bag_container_variants(self):
        two = [backpack(165, 225), bag("suitcase", [300, 300, 340, 400])]  # daypack + 100 px suitcase
        for container in (two, tuple(two), (b for b in two), np.array(two, dtype=object)):
            with self.subTest(container=type(container).__name__):
                self.assertIs(classify_person_bags(PERSON, keypoints(), container)["large_bag"], True)
        for invalid in (5, "backpack", backpack(165, 225)):
            with self.subTest(invalid=invalid):
                result = classify_person_bags(PERSON, keypoints(), invalid)
                self.assertIsNone(result["large_bag"])
                self.assertIn("bags_input_invalid_ignored", result["reasons"])
        mixed = classify_person_bags(PERSON, keypoints(), [None, 3, backpack(165, 225)])
        self.assertIs(mixed["large_bag"], False)
        self.assertIn("bag_record_invalid_ignored", mixed["reasons"])
        self.assertEqual(mixed["features"]["bags"][0]["index"], 2)  # index refers to the input list

    def test_assessable_flag_that_cannot_be_read_counts_as_not_assessable(self):
        result = classify_person_bags(PERSON, keypoints(), [], assessable=np.array([True, False]))
        self.assertIsNone(result["large_bag"])
        self.assertIn("assessable_flag_invalid", result["reasons"])
        self.assertIs(result["features"]["assessable"], False)

    def test_label_normalisation_collapses_whitespace_and_case(self):
        for label in ("  DUFFEL   bag ", "duffel_bag", "Duffel\tBag", "SUITCASE"):
            with self.subTest(label=label):
                self.assertIs(classify_person_bags(PERSON, None, [bag(label, [0, 0, 10, 10])])["large_bag"], True)
        for label in (None, 3, "duffle bag", "back pack", "handbag", ""):
            with self.subTest(label=label):
                result = classify_person_bags(PERSON, None, [bag(label, [0, 0, 10, 10])])
                self.assertEqual(result["features"]["bags"][0]["rule"], "unsupported_label_ignored")

    def test_boxes_partly_outside_the_image_and_oversized_bags(self):
        # Negative coordinates are legal (box clipped by the frame edge upstream or not at all).
        shifted = [[x, y - 120.0, c] for x, y, c in keypoints()]  # shoulders 40, hips 140
        self.assertEqual(verdict([-50, -20, 50, 280], shifted, [backpack(-35, 105)]),
                         (True, "backpack_tall_and_above_shoulders_large"))
        self.assertIs(classify_person_bags(PERSON, keypoints(), [backpack(0, 600)])["large_bag"], True)

    def test_duplicate_iou_boundary_is_inclusive(self):
        base = bag("suitcase", [0, 0, 10, 10], .9)
        half = bag("suitcase", [0, 0, 10, 5], .8)       # IoU = 50 / 100 = .5
        less = bag("suitcase", [0, 0, 10, 4.9], .8)     # IoU = .49
        self.assertEqual(classify_person_bags(PERSON, None, [base, half])["features"]["large_bag_count"], 1)
        self.assertEqual(classify_person_bags(PERSON, None, [base, less])["features"]["large_bag_count"], 2)
        self.assertEqual(count_large_bags([], [base, half])["large_bags"], 1)
        self.assertEqual(count_large_bags([], [base, less])["large_bags"], 2)
        touching = [bag("suitcase", [0, 0, 10, 10]), bag("suitcase", [10, 0, 20, 10])]  # shared edge only
        self.assertEqual(count_large_bags([], touching)["large_bags"], 2)

    def test_policy_is_read_at_call_time(self):
        slightly_above = [backpack(144, 244)]  # F1 1.00, F2 .16
        self.assertIsNone(classify_person_bags(PERSON, keypoints(), slightly_above)["large_bag"])
        with mock.patch.dict(POLICY, {"daypack_top_above_shoulder": .16}):
            self.assertIs(classify_person_bags(PERSON, keypoints(), slightly_above)["large_bag"], False)
        small_suitcase = [bag("suitcase", [210, 330, 250, 400])]  # long side 70
        self.assertIsNone(classify_person_bags(PERSON, keypoints(), small_suitcase)["large_bag"])
        with mock.patch.dict(POLICY, {"luggage_min_torso_ratio": .5}):
            self.assertIs(classify_person_bags(PERSON, keypoints(), small_suitcase)["large_bag"], True)
        self.assertIsNone(classify_person_bags(PARTIAL, keypoints(), [backpack(170, 320)])["large_bag"])
        with mock.patch.dict(POLICY, {"full_body_per_torso": 2.5}):
            self.assertIs(classify_person_bags(PARTIAL, keypoints(), [backpack(170, 320)])["large_bag"], True)
        with mock.patch.dict(POLICY, {"keypoint_confidence": .95}):
            self.assertIsNone(classify_person_bags(PERSON, keypoints(), [backpack(135, 275)])["large_bag"])
        with mock.patch.dict(POLICY, {"luggage_confidence": .9}):
            self.assertEqual(count_large_bags([], [bag("suitcase", [0, 0, 5, 5], .8)])["large_bags"], 0)

    def test_equal_confidence_duplicates_do_not_depend_on_input_order(self):
        # a~b and b~c overlap, a and c do not: greedy merging must not follow input order.
        a, b, c = (bag("suitcase", [x, 0, x + 10, 10], .6) for x in (0, 3, 6))
        counts = {classify_person_bags(PERSON, None, list(order))["features"]["large_bag_count"]
                  for order in ([a, b, c], [c, b, a], [b, a, c], [b, c, a])}
        self.assertEqual(len(counts), 1)
        totals = {tuple(count_large_bags([], list(order)).values()) for order in ([a, b, c], [c, b, a], [b, a, c])}
        self.assertEqual(len(totals), 1)

    def test_module_is_pure_python_and_device_independent(self):
        # Post-processing only: identical on CUDA and CPU-only machines, no numpy/torch in the module.
        tree = ast.parse(Path(bags_module.__file__).read_text(encoding="utf-8"))
        imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertLessEqual(imported, {"__future__", "math", "operator"})


class CountLargeBagsAdversarialTests(unittest.TestCase):
    def test_malformed_results_do_not_crash(self):
        people = [{"large_bag": True, "features": [1, 2]}, {"large_bag": True, "features": "x"},
                  {"large_bag": True, "features": {"bags": "junk", "large_bag_count": -3}},
                  {"large_bag": None, "features": {"bags": [None, {"status": "large", "box": None},
                                                           {"status": ["large"], "box": [0, 0, 5, 5]}]}},
                  {"large_bag": "yes"}, {"large_bag": 1}, {}, []]
        self.assertEqual(count_large_bags(people, []), {"large_bags": 3, "large_bags_uncertain": 0})
        for junk in (5, "abc", {"large_bag": True}):
            with self.subTest(junk=junk):
                self.assertEqual(count_large_bags(junk, junk), {"large_bags": 0, "large_bags_uncertain": 0})

    def test_numpy_states_and_counts_are_honoured(self):
        people = [{"large_bag": np.bool_(True)},
                  {"large_bag": np.True_, "features": {"large_bag_count": np.int64(2), "uncertain_bag_count": np.int32(1)}},
                  {"large_bag": np.bool_(False), "features": {"large_bag_count": np.int64(4)}}]
        counts = count_large_bags(np.array(people, dtype=object), [])
        self.assertEqual(counts, {"large_bags": 3, "large_bags_uncertain": 1})
        self.assertTrue(all(type(v) is int for v in counts.values()))

    def test_loose_luggage_already_carried_is_not_counted_twice(self):
        carried = classify_person_bags(PERSON, None, [bag("suitcase", [210, 330, 250, 400])])
        same = bag("suitcase", [211, 331, 250, 400], .6)
        other = bag("duffel bag", [600, 330, 650, 400], .6)
        self.assertEqual(count_large_bags([carried], [same]), {"large_bags": 1, "large_bags_uncertain": 0})
        self.assertEqual(count_large_bags([carried], [same, other]), {"large_bags": 2, "large_bags_uncertain": 0})

    def test_confident_loose_luggage_upgrades_the_carried_reading(self):
        uncertain = classify_person_bags(PERSON, keypoints(), [backpack(160, 290)])
        daypack = classify_person_bags(PERSON, keypoints(), [backpack(165, 225)])
        self.assertEqual(count_large_bags([uncertain], []), {"large_bags": 0, "large_bags_uncertain": 1})
        on_uncertain = [bag("suitcase", [121, 161, 180, 290], .5), bag("suitcase", [120, 160, 181, 291], .4)]
        self.assertEqual(count_large_bags([uncertain], on_uncertain), {"large_bags": 1, "large_bags_uncertain": 0})
        on_daypack = [bag("duffel bag", [120, 165, 180, 225], .5)]
        self.assertEqual(count_large_bags([daypack], on_daypack), {"large_bags": 1, "large_bags_uncertain": 0})
        # Low-confidence loose luggage never touches the carried reading.
        self.assertEqual(count_large_bags([uncertain], [bag("suitcase", [121, 161, 180, 290], .2)]),
                         {"large_bags": 0, "large_bags_uncertain": 1})

    def test_carried_match_does_not_depend_on_person_order(self):
        first = classify_person_bags(PERSON, keypoints(), [backpack(160, 290)])                   # uncertain
        second = classify_person_bags(PERSON, None, [bag("suitcase", [122, 162, 182, 292])])      # large
        self.assertEqual((first["large_bag"], second["large_bag"]), (None, True))
        loose = [bag("suitcase", [121, 161, 181, 291], .7)]  # overlaps both equally
        expected = {"large_bags": 1, "large_bags_uncertain": 1}
        self.assertEqual(count_large_bags([first, second], loose), expected)
        self.assertEqual(count_large_bags([second, first], loose), expected)

    def test_duplicate_carried_entries_are_not_used_twice(self):
        person = classify_person_bags(PERSON, keypoints(), [backpack(160, 290), backpack(161, 290, confidence=.5)])
        self.assertEqual(person["features"]["uncertain_bag_count"], 1)
        loose = [bag("suitcase", [120, 160, 180, 290], .9)]
        self.assertEqual(count_large_bags([person], loose), {"large_bags": 1, "large_bags_uncertain": 0})

    def test_inputs_are_not_mutated(self):
        people = [classify_person_bags(PERSON, keypoints(), [backpack(160, 290)])]
        loose = [bag("suitcase", [121, 161, 180, 290], .5)]
        before = copy.deepcopy((people, loose))
        count_large_bags(people, loose)
        self.assertEqual((people, loose), before)


def random_case(rng):
    """One synthetic person with jittered keypoints and 0-6 (partly duplicated) bags."""
    x, y = rng.uniform(-50, 800), rng.uniform(-50, 500)
    width, height = rng.uniform(10, 200), rng.uniform(40, 600)
    person = rng.choice([[x, y, x + width, y + height]] * 8 + [None, [x, y, x, y + height]])
    points = None
    if rng.random() < .8:
        shoulder, hip = y + height * rng.uniform(.1, .35), y + height * rng.uniform(.2, .65)
        points = [[x + rng.uniform(0, width), y + rng.uniform(0, height), rng.random()] for _ in range(17)]
        for index, row_y in ((5, shoulder), (6, shoulder), (11, hip), (12, hip)):
            points[index] = [x + rng.uniform(0, width), row_y + rng.uniform(-8, 8),
                             rng.choice([rng.random(), .9, .5, math.nan])]
    confidences = iter(rng.sample(range(100, 1000), 12))
    detections = []
    for _ in range(rng.randint(0, 4)):
        top = y + height * rng.uniform(-.2, .8)
        box = [x + rng.uniform(-10, width), top, 0, top + height * rng.uniform(.02, .6)]
        box[2] = box[0] + rng.uniform(1, 80)
        detections.append({"label": rng.choice(["backpack"] * 4 + ["suitcase", "duffel bag", "handbag"]),
                           "box": box, "confidence": next(confidences) / 1000})
        if rng.random() < .4:  # a second expert's reading of the same bag
            twin = [v + rng.uniform(-3, 3) for v in box]
            detections.append({"label": rng.choice(["backpack", "suitcase"]), "xyxy": twin,
                               "confidence": next(confidences) / 1000})
    return person, points, detections


def expected_backpack_verdict(entry):
    """bags_v2 verdict re-derived from the reported (rounded) cues."""
    f1, f2, f3 = entry["torso_ratio"], entry["top_above_shoulder_per_torso"], entry["height_ratio"]
    if f1 is not None and f2 is not None and f1 >= POLICY["large_torso_ratio"] \
            and f2 >= POLICY["large_top_above_shoulder"]:
        return True
    if entry["full_body"] and f3 is not None and f3 >= POLICY["large_height_ratio"]:
        return True
    if f1 is not None:
        daypack = f1 <= POLICY["daypack_torso_ratio"] and (f2 is None or f2 <= POLICY["daypack_top_above_shoulder"])
        return False if daypack else None
    return False if f3 is not None and f3 <= POLICY["daypack_height_ratio"] else None


class PropertyTests(unittest.TestCase):
    """Seeded fuzzing: invariants that must hold for any input."""

    def check_backpack_entry(self, entry, features):
        expected = expected_backpack_verdict(entry)
        self.assertIs(entry["verdict"], expected, entry)
        self.assertIn(entry["rule"], BACKPACK_RULES[expected], entry)
        torso, person_height = features["torso_px"], features["person_height_px"]
        self.assertEqual(entry["torso_ratio"] is None, torso is None, entry)
        if entry["full_body"]:
            self.assertIsNotNone(torso)
            self.assertGreaterEqual(person_height, POLICY["full_body_per_torso"] * torso - 1e-4)
        elif torso is not None and person_height is not None:
            self.assertLessEqual(person_height, POLICY["full_body_per_torso"] * torso + 1e-4)

    def check_luggage_entry(self, entry, features):
        box, torso = entry["box"], features["torso_px"]
        small = torso is not None and max(box[2] - box[0], box[3] - box[1]) < POLICY["luggage_min_torso_ratio"] * torso
        self.assertEqual(entry["status"], "uncertain" if small else "large", entry)
        self.assertTrue(entry["rule"].endswith("_small_for_luggage" if small else "_confident_large"), entry)
        self.assertGreaterEqual(entry["confidence"], POLICY["luggage_confidence"])

    def test_random_inputs_satisfy_invariants(self):
        rng = random.Random(20260923)
        results, loose, all_luggage = [], [], []
        for case in range(400):
            person, points, detections = random_case(rng)
            before = repr((person, points, detections))
            result = classify_person_bags(person, points, detections, assessable=case % 3 == 0)
            with self.subTest(case=case):
                self.assertEqual(repr((person, points, detections)), before)
                strict_json(self, result)
                self.assertEqual(classify_person_bags(person, points, detections, assessable=case % 3 == 0), result)
                features = result["features"]
                expected = (True if features["large_bag_count"] else None if features["uncertain_bag_count"]
                            else False if features["daypack_count"] or case % 3 == 0 else None)
                self.assertIs(result["large_bag"], expected)
                for entry in features["bags"]:
                    if entry["status"] == "ignored":
                        continue
                    if entry["label"] == "backpack":
                        self.check_backpack_entry(entry, features)
                    else:
                        self.check_luggage_entry(entry, features)
                shuffled = detections[:]
                rng.shuffle(shuffled)
                again = classify_person_bags(person, points, shuffled, assessable=case % 3 == 0)
                self.assertIs(again["large_bag"], result["large_bag"])
                self.assertEqual([again["features"][k] for k in ("large_bag_count", "uncertain_bag_count", "daypack_count")],
                                 [features[k] for k in ("large_bag_count", "uncertain_bag_count", "daypack_count")])
                self.assertEqual(set(again["reasons"]), set(result["reasons"]))
            results.append(result)
            small = {e["index"] for e in result["features"]["bags"] if str(e["rule"]).endswith("_small_for_luggage")}
            for index, detection in enumerate(detections):
                if detection["label"] != "backpack":
                    all_luggage.append(detection)
                    if index not in small:
                        loose.append(detection)
        totals = count_large_bags(results, loose)
        strict_json(self, totals)
        for _ in range(5):
            rng.shuffle(results)
            rng.shuffle(loose)
            self.assertEqual(count_large_bags(results, loose), totals)
        carried_only = count_large_bags(results, [])
        self.assertEqual(carried_only["large_bags"], sum(r["features"]["large_bag_count"] for r in results))
        self.assertEqual(carried_only["large_bags_uncertain"], sum(r["features"]["uncertain_bag_count"] for r in results))
        # Every "loose" item here is a carried luggage detection: none may be counted twice.
        self.assertEqual(totals, carried_only)
        # Passing every luggage detection (also those ruled small for luggage) never
        # adds bags: at most an uncertain carried reading is settled as large.
        everything = count_large_bags(results, all_luggage)
        self.assertGreaterEqual(everything["large_bags"], carried_only["large_bags"])
        self.assertLessEqual(sum(everything.values()), sum(carried_only.values()))
        # The same luggage moved away from every person is genuinely loose: purely additive.
        away = [{**item, "box": None, "xyxy": [v + 10000 for v in item.get("box") or item["xyxy"]]}
                for item in all_luggage]
        loose_only = count_large_bags([], away)["large_bags"]
        self.assertGreater(loose_only, 20)
        self.assertEqual(count_large_bags(results, away),
                         {"large_bags": carried_only["large_bags"] + loose_only,
                          "large_bags_uncertain": carried_only["large_bags_uncertain"]})

    def test_random_inputs_exercise_every_backpack_rule(self):
        # Guards the fuzzing above against silently covering only a few branches.
        rng = random.Random(20260923)
        seen = set()
        for _ in range(400):
            person, points, detections = random_case(rng)
            for entry in classify_person_bags(person, points, detections)["features"]["bags"]:
                seen.add(entry["rule"])
        wanted = set().union(*BACKPACK_RULES.values()) | {"suitcase_small_for_luggage", "suitcase_confident_large"}
        self.assertLessEqual(wanted, seen)

    def test_large_inputs_stay_fast(self):
        # 300 = the detectors' max_det, far above a real trail-camera frame.
        rng = random.Random(3)
        detections = [bag(rng.choice(["backpack", "suitcase", "duffel bag"]),
                          [x := rng.uniform(0, 4000), y := rng.uniform(0, 3000), x + rng.uniform(5, 80), y + rng.uniform(5, 120)],
                          rng.uniform(.1, 1)) for _ in range(300)]
        started = time.perf_counter()
        result = classify_person_bags([0, 0, 4000, 3000], keypoints(), detections)
        people = [classify_person_bags(PERSON, keypoints(), detections[i:i + 6]) for i in range(0, 300, 6)]
        totals = count_large_bags(people + [result], detections)
        self.assertLess(time.perf_counter() - started, 3.0)
        self.assertEqual(len(result["features"]["bags"]), 300)
        strict_json(self, totals)


if __name__ == "__main__":
    unittest.main()
