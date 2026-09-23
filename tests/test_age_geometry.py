"""Synthetic geometry invariants; these do not measure age accuracy on real people."""
import copy
import json
import math
import random
import time
import tracemalloc
import unittest
import warnings
from unittest import mock

import numpy as np

from trailcam.age_geometry import POLICY, classify, fit_camera, person_geometry

W, H = 1000, 800


def skeleton():
    """Upright person: eyes y=125, shoulders 170, hips 260, ankles 390."""
    points = [[130.0, 250.0, 0.1] for _ in range(17)]
    for index, point in {0: (130, 130), 1: (135, 125), 2: (125, 125), 3: (140, 128), 4: (120, 128),
                         5: (145, 170), 6: (115, 170), 7: (150, 215), 8: (110, 215),
                         9: (152, 255), 10: (108, 255), 11: (140, 260), 12: (120, 260),
                         13: (138, 330), 14: (122, 330), 15: (137, 390), 16: (123, 390)}.items():
        points[index] = [float(point[0]), float(point[1]), 0.9]
    return points


DEFAULT = object()


def person(keypoints=DEFAULT, box=(100, 100, 160, 400), **extra):
    return {"person_id": "p1", "image_id": "img1", "camera_id": "cam",
            "box": list(box) if isinstance(box, tuple) else box,
            "keypoints": skeleton() if keypoints is DEFAULT else keypoints,
            "image_width": W, "image_height": H, "strip_bottom": 0, **extra}


def geom(h, foot_y, foot_x=500.0, pid="p", image="img", complete=True, image_height=1000):
    return {"person_id": pid, "image_id": image, "camera_id": "cam", "image_height": image_height,
            "height_px": h, "foot_x": foot_x, "foot_y": foot_y, "complete": complete, "reasons": []}


def synthetic_camera(seed, n=150, children=.30, sitting=.10, noise=.05, b=0.0):
    """Adults h = 0.35*y + b*x - 200 with multiplicative noise; children x0.65, sitting x0.6."""
    rng = np.random.RandomState(seed)
    nc, ns = round(n * children), round(n * sitting)
    kinds = ["child"] * nc + ["sitting"] * ns + ["adult"] * (n - nc - ns)
    kinds = [kinds[i] for i in rng.permutation(n)]
    geoms = []
    for i, kind in enumerate(kinds):
        y, x = float(rng.uniform(900, 1450)), float(rng.uniform(50, 2000))
        h = (0.35 * y + b * x - 200) * {"child": .65, "sitting": .6, "adult": 1.0}[kind] * (1 + noise * rng.randn())
        geoms.append(geom(round(h, 2), round(y, 2), round(x, 2), "p%d" % i, "img%d" % i, image_height=1500))
    return geoms, kinds


def mean_prediction_error(calibration, geoms, b=0.0):
    a, bb, c = calibration["coef"]
    ys = np.linspace(min(g["foot_y"] for g in geoms), max(g["foot_y"] for g in geoms), 15)
    xs = np.linspace(min(g["foot_x"] for g in geoms), max(g["foot_x"] for g in geoms), 15)
    Y, X = np.meshgrid(ys, xs)
    return float(np.mean(np.abs((a * Y + bb * X + c) / (0.35 * Y + b * X - 200) - 1)))


def plain(value):
    """True when value holds only JSON-native Python types (no numpy scalars)."""
    if isinstance(value, dict):
        return all(isinstance(k, str) and plain(v) for k, v in value.items())
    if isinstance(value, list):
        return all(plain(v) for v in value)
    return value is None or type(value) in (bool, int, float, str)


class PersonGeometryTests(unittest.TestCase):
    def test_upright_person_is_complete_and_crown_extends_above_eyes(self):
        g = person_geometry(person())
        self.assertTrue(g["complete"], g["reasons"])
        self.assertEqual(g["reasons"], [])
        # crown = 125 - 0.5 * (170 - 125) = 102.5; foot = lowest ankle 390
        self.assertEqual((g["height_px"], g["foot_x"], g["foot_y"]), (287.5, 130.0, 390.0))
        self.assertEqual((g["head_source"], g["foot_source"]), ("keypoint_crown", "ankles"))
        self.assertEqual((g["person_id"], g["image_id"], g["camera_id"], g["image_height"]),
                         ("p1", "img1", "cam", 800.0))
        self.assertTrue(plain(g))
        json.dumps(g)

    def test_lowest_confident_ankle_is_the_foot(self):
        kps = skeleton()
        kps[15] = [150.0, 395.0, 0.9]
        kps[16] = [110.0, 380.0, 0.9]
        g = person_geometry(person(kps))
        self.assertEqual((g["foot_x"], g["foot_y"], g["height_px"]), (130.0, 395.0, 292.5))

    def test_ears_stand_in_for_eyes_and_face_only_falls_back_to_box_top(self):
        kps = skeleton()
        kps[1][2] = kps[2][2] = 0.1
        g = person_geometry(person(kps))
        # head top = ear y 128 (nose 130); crown = 128 - 0.5 * (170 - 128) = 107
        self.assertEqual((g["height_px"], g["head_source"]), (283.0, "keypoint_crown"))
        self.assertTrue(g["complete"])
        for i in (3, 4):
            kps[i][2] = 0.1
        g = person_geometry(person(kps))
        self.assertEqual((g["height_px"], g["head_source"]), (290.0, "box_top"))
        self.assertTrue(g["complete"])

    def test_keypoint_confidence_boundary(self):
        kps = skeleton()
        kps[15][2] = kps[16][2] = POLICY["keypoint_confidence"]
        self.assertTrue(person_geometry(person(kps))["complete"])
        kps[15][2] = kps[16][2] = 0.399
        g = person_geometry(person(kps))
        self.assertFalse(g["complete"])
        self.assertEqual(g["reasons"], ["no_confident_ankle"])
        self.assertEqual((g["foot_source"], g["foot_y"], g["foot_x"], g["height_px"]),
                         ("box_bottom", 400.0, 130.0, 297.5))

    def test_missing_head_keypoints_are_incomplete(self):
        kps = skeleton()
        for i in range(5):
            kps[i][2] = 0.0
        g = person_geometry(person(kps))
        self.assertFalse(g["complete"])
        self.assertIn("no_head_keypoints", g["reasons"])
        self.assertEqual((g["head_source"], g["height_px"]), ("box_top", 290.0))

    def test_no_keypoints_uses_box_and_is_incomplete(self):
        g = person_geometry(person(None))
        self.assertEqual(g["reasons"], ["no_keypoints"])
        self.assertFalse(g["complete"])
        self.assertEqual((g["height_px"], g["foot_x"], g["foot_y"]), (300.0, 130.0, 400.0))
        self.assertEqual((g["head_source"], g["foot_source"]), ("box_top", "box_bottom"))

    def test_malformed_keypoints_are_treated_as_absent(self):
        nan = skeleton()
        nan[3][0] = float("nan")
        short_row = skeleton()
        short_row[4] = [1.0, 2.0]
        for bad in (skeleton()[:16], nan, short_row, "keypoints", [[None, None, None]] * 17, 5):
            g = person_geometry(person(bad))
            self.assertEqual(g["reasons"], ["invalid_keypoints"], bad)
            self.assertFalse(g["complete"])
            self.assertEqual(g["height_px"], 300.0)

    def test_degenerate_boxes_yield_no_geometry(self):
        for box in (None, [], [1, 2, 3], [100, 100, 100, 400], [100, 400, 160, 100],
                    [160, 100, 100, 400], [0, 0, float("nan"), 10], [0, 0, float("inf"), 10],
                    "abcd", [True, 0, 10, 10], 7):
            g = person_geometry(person(box=box))
            self.assertEqual(g["reasons"], ["invalid_box"], box)
            self.assertFalse(g["complete"])
            self.assertIsNone(g["height_px"])
            self.assertIsNone(g["foot_y"])
        for junk in (None, {}, "person"):
            self.assertEqual(person_geometry(junk)["reasons"], ["invalid_box"])

    def test_numpy_inputs_give_plain_outputs(self):
        p = person(np.array(skeleton(), dtype=np.float32), box=tuple(np.array([100, 100, 160, 400], dtype=np.float64)))
        p.update(image_width=np.int64(W), image_height=np.int32(H), strip_bottom=np.int64(0))
        g = person_geometry(p)
        self.assertTrue(g["complete"], g["reasons"])
        self.assertTrue(plain(g))

    def test_border_margins_are_inclusive(self):
        m = POLICY["border_margin_px"]
        cases = [((m, 100, 160, 400), "touches_left_border"),
                 ((100, 100, W - m, 400), "touches_right_border"),
                 ((100, m, 160, 400), "touches_top_border"),
                 ((100, 100, 160, H - m), "touches_bottom_or_data_strip")]
        for box, code in cases:
            g = person_geometry(person(box=box))
            self.assertEqual(g["reasons"], [code], box)
            self.assertFalse(g["complete"])
        for box in ((m + .5, 100, 160, 400), (100, 100, W - m - .5, 400),
                    (100, m + .5, 160, 400), (100, 100, 160, H - m - .5)):
            self.assertTrue(person_geometry(person(box=box))["complete"], box)

    def test_data_strip_counts_as_bottom_border(self):
        strip = 40
        touching = person(box=(100, 100, 160, H - strip - 2), strip_bottom=strip)
        self.assertEqual(person_geometry(touching)["reasons"], ["touches_bottom_or_data_strip"])
        clear = person(box=(100, 100, 160, H - strip - 2.5), strip_bottom=strip)
        self.assertTrue(person_geometry(clear)["complete"])
        for strip_value in (None, -30, "bad"):
            self.assertTrue(person_geometry(person(strip_bottom=strip_value))["complete"])

    def test_invalid_image_dimensions_prevent_completeness(self):
        for width, height in ((0, H), (W, None), (-5, H), (W, float("nan"))):
            g = person_geometry(person(image_width=width, image_height=height))
            self.assertIn("invalid_image_dimensions", g["reasons"])
            self.assertFalse(g["complete"])
            self.assertEqual(g["height_px"], 287.5)

    def test_torso_requires_shoulder_and_hip(self):
        kps = skeleton()
        kps[11][2] = kps[12][2] = 0.1
        self.assertEqual(person_geometry(person(kps))["reasons"], ["no_torso_keypoints"])
        kps = skeleton()
        kps[5][2] = kps[6][2] = 0.1
        # without shoulders the crown cannot be extended either
        g = person_geometry(person(kps))
        self.assertEqual((g["reasons"], g["head_source"]), (["no_torso_keypoints"], "box_top"))
        kps = skeleton()
        kps[6][2] = kps[12][2] = 0.1  # side view: one shoulder and one hip suffice
        self.assertTrue(person_geometry(person(kps))["complete"])

    def test_upright_torso_boundary(self):
        minimum = POLICY["min_torso_per_height"] * 287.5  # 43.125
        kps = skeleton()
        kps[11][1] = kps[12][1] = 170 + minimum
        self.assertTrue(person_geometry(person(kps))["complete"])
        kps[11][1] = kps[12][1] = 170 + minimum - 0.01
        self.assertEqual(person_geometry(person(kps))["reasons"], ["torso_not_upright"])
        kps[11][1] = kps[12][1] = 160  # hips above shoulders: lying / inverted
        self.assertEqual(person_geometry(person(kps))["reasons"], ["torso_not_upright"])

    def test_seated_person_is_not_a_standing_height_observation(self):
        kps = skeleton()
        for i, y in ((13, 262), (14, 262), (15, 330), (16, 330)):  # thighs horizontal
            kps[i][1] = y
        g = person_geometry(person(kps))
        self.assertEqual(g["reasons"], ["legs_not_extended"])
        self.assertFalse(g["complete"])
        # 1.20 (was 1.10): a seated adult who still passes then measures well above the 0.82 child cut.
        self.assertEqual(POLICY["min_leg_per_torso"], 1.20)
        limit = 260 + POLICY["min_leg_per_torso"] * 90  # hips 260, torso 90
        for foot, complete in ((limit + .01, True), (limit - .01, False)):
            kps = skeleton()
            kps[15][1] = kps[16][1] = foot
            self.assertEqual(person_geometry(person(kps))["complete"], complete, foot)

    def test_nonpositive_height_is_rejected(self):
        kps = skeleton()
        kps[15][1] = kps[16][1] = 90.0  # ankles above the crown
        g = person_geometry(person(kps))
        self.assertEqual(g["reasons"], ["nonpositive_height"])
        self.assertIsNone(g["height_px"])
        self.assertFalse(g["complete"])

    def test_input_is_not_mutated_and_result_is_deterministic(self):
        p = person()
        before = copy.deepcopy(p)
        self.assertEqual(person_geometry(p), person_geometry(p))
        self.assertEqual(p, before)


class FitCameraTests(unittest.TestCase):
    def test_recovers_synthetic_camera_despite_children_and_sitting_people(self):
        for seed in range(5):
            geoms, kinds = synthetic_camera(seed)
            cal = fit_camera(geoms)
            self.assertEqual((cal["status"], cal["model"], cal["n"]), ("ok", "linear_xy", 150), seed)
            self.assertLessEqual(mean_prediction_error(cal, geoms), .03, seed)
            a, b, c = cal["coef"]
            centre = (1175.0, 1025.0)
            self.assertAlmostEqual((a * centre[0] + b * centre[1] + c) / (0.35 * centre[0] - 200), 1.0, delta=.03)
            self.assertLess(cal["residual_mad"], .05)
            self.assertGreaterEqual(cal["inliers"], 80)  # ~90 adults; children/sitting rejected
            self.assertLessEqual(cal["inliers"], 100)
            labels = [classify(g, cal, [])["age"] for g in geoms]
            for kind in ("adult", "child"):
                hits = [label == kind for label, truth in zip(labels, kinds) if truth == kind]
                self.assertGreaterEqual(sum(hits) / len(hits), .90, (seed, kind))
            self.assertTrue(plain(cal))
            json.dumps(cal)

    def test_tolerates_up_to_35_percent_children_plus_sitting(self):
        for seed in range(3):
            geoms, _ = synthetic_camera(seed, children=.35, sitting=.10)
            cal = fit_camera(geoms)
            self.assertEqual(cal["status"], "ok", seed)
            self.assertLessEqual(mean_prediction_error(cal, geoms), .03, seed)

    def test_lateral_term_is_recovered(self):
        geoms, _ = synthetic_camera(7, n=200, b=0.03)
        cal = fit_camera(geoms)
        self.assertEqual((cal["status"], cal["model"]), ("ok", "linear_xy"))
        self.assertAlmostEqual(cal["coef"][1], 0.03, delta=.01)
        self.assertLessEqual(mean_prediction_error(cal, geoms, b=0.03), .03)

    def test_fit_is_order_invariant_and_deterministic(self):
        geoms, _ = synthetic_camera(3)
        shuffled = [geoms[i] for i in np.random.RandomState(99).permutation(len(geoms))]
        self.assertEqual(fit_camera(geoms), fit_camera(shuffled))
        self.assertEqual(fit_camera(geoms), fit_camera(geoms))

    def test_insufficient_people(self):
        self.assertEqual(fit_camera([])["status"], "insufficient")
        self.assertEqual(fit_camera(None)["n"], 0)
        geoms, _ = synthetic_camera(0, n=19, children=0, sitting=0)
        incomplete = [dict(g, complete=False) for g in synthetic_camera(1, n=50)[0]]
        junk = [None, "x", geom(None, 1000.0), geom(-5.0, 1000.0), geom(100.0, float("nan")),
                {k: v for k, v in geom(100.0, 1000.0).items() if k != "complete"}]
        result = fit_camera(geoms + incomplete + junk)
        self.assertEqual((result["status"], result["n"], result["coef"]), ("insufficient", 19, None))
        self.assertEqual(result["reasons"], ["fewer_complete_people_than_minimum"])
        self.assertEqual(fit_camera(geoms, min_people=19)["status"], "ok")
        few = synthetic_camera(0, n=4, children=0, sitting=0)[0]
        self.assertEqual(fit_camera(few, min_people=0)["status"], "insufficient")  # floor of 5
        self.assertTrue(plain(result))

    def test_same_foot_y_falls_back_to_constant_height_when_consistent(self):
        rng = np.random.RandomState(4)
        geoms = [geom(round(200 * (.65 if i % 10 < 3 else 1) * (1 + .04 * rng.randn()), 2), 1000.0,
                      float(rng.uniform(0, 2000)), "p%d" % i) for i in range(60)]
        cal = fit_camera(geoms)
        self.assertEqual((cal["status"], cal["model"]), ("ok", "constant"))
        self.assertEqual(cal["coef"][:2], [0.0, 0.0])
        self.assertAlmostEqual(cal["coef"][2] / 200, 1.0, delta=.03)
        self.assertEqual(cal["foot_y_range"], [1000.0, 1000.0])
        self.assertEqual(classify(geom(130.0, 1000.0), cal, [])["age"], "child")
        self.assertEqual(classify(geom(198.0, 1000.0), cal, [])["age"], "adult")

    def test_same_foot_y_with_scattered_heights_is_unreliable(self):
        rng = np.random.RandomState(5)
        # +-15% spread: fine for a sloped fit (MAD ~.075 <= .10) but not for the constant fallback
        heights = [round(float(v), 2) for v in rng.uniform(170, 230, 60)]
        flat = fit_camera([geom(h, 1000.0, 20.0 * i) for i, h in enumerate(heights)])
        self.assertEqual((flat["status"], flat["model"]), ("unreliable", "constant"))
        self.assertIn("constant_model_residual_too_large", flat["reasons"])
        self.assertGreater(flat["residual_mad"], POLICY["max_residual_mad_constant"])
        wild = fit_camera([geom(round(float(v), 2), 1000.0) for v in rng.uniform(60, 300, 60)])
        self.assertEqual((wild["status"], wild["model"]), ("unreliable", "constant"))

    def test_degenerate_or_collinear_foot_x_drops_the_lateral_term(self):
        rng = np.random.RandomState(6)
        ys = rng.uniform(900, 1450, 40)
        same_x = [geom(round((0.35 * y - 200) * (1 + .03 * rng.randn()), 2), float(y), 700.0) for y in ys]
        collinear = [dict(g, foot_x=2 * g["foot_y"] + 1) for g in same_x]
        for geoms in (same_x, collinear):
            cal = fit_camera(geoms)
            self.assertEqual((cal["status"], cal["model"], cal["coef"][1]), ("ok", "linear_y", 0.0))
            self.assertAlmostEqual(cal["coef"][0], 0.35, delta=.02)

    def test_unstructured_heights_are_unreliable(self):
        rng = np.random.RandomState(8)
        geoms = [geom(round(float(rng.uniform(50, 400)), 2), float(rng.uniform(800, 1400)),
                      float(rng.uniform(0, 2000))) for _ in range(60)]
        self.assertEqual(fit_camera(geoms)["status"], "unreliable")

    def test_no_dominant_height_mode_is_unreliable(self):
        rng = np.random.RandomState(9)
        geoms = [geom(round((0.35 * y - 200) * f * (1 + .03 * rng.randn()), 2), float(y), float(rng.uniform(0, 2000)))
                 for f in (1.0, .7, .45) for y in rng.uniform(900, 1450, 30)]
        cal = fit_camera(geoms)
        self.assertEqual(cal["status"], "unreliable")
        self.assertIn("low_inlier_fraction", cal["reasons"])

    def test_nonpositive_prediction_in_observed_range_is_unreliable(self):
        rng = np.random.RandomState(10)
        adults = [geom(round((0.4 * y - 200) * (1 + .03 * rng.randn()), 2), float(y)) for y in rng.uniform(600, 1200, 60)]
        tiny_far = [geom(20.0, float(y)) for y in rng.uniform(300, 400, 30)]  # plane predicts <= 0 here
        cal = fit_camera(adults + tiny_far)
        self.assertEqual(cal["status"], "unreliable")
        self.assertEqual(cal["reasons"], ["nonpositive_prediction_in_range"])
        self.assertAlmostEqual(cal["coef"][0], 0.4, delta=.02)
        self.assertEqual(classify(geom(200.0, 1000.0, image_height=1500), cal, [])["method"], "none")

    def test_exact_data_has_zero_residual(self):
        geoms = [geom(0.35 * y - 200, float(y), 50.0 + 7 * i) for i, y in enumerate(range(900, 1450, 20))]
        cal = fit_camera(geoms)
        self.assertEqual((cal["status"], cal["inliers"], cal["residual_mad"]), ("ok", len(geoms), 0.0))
        self.assertEqual(cal["coef"], [0.35, 0.0, -200.0])


def calibration(c=100.0, y_range=(400.0, 600.0), status="ok", coef=None):
    return {"status": status, "coef": coef if coef is not None else [0.0, 0.0, c],
            "foot_y_range": list(y_range), "foot_x_range": [0.0, 1000.0]}


class ClassifyTests(unittest.TestCase):
    def test_calibrated_thresholds_are_inclusive(self):
        # The calibrated child cut is 0.82 (it was 0.78 before validation tuning;
        # see POLICY["calibration_child_max"]); the adult cut stays 0.88.
        self.assertEqual((POLICY["calibration_child_max"], POLICY["calibration_adult_min"]), (0.82, 0.88))
        cal = calibration()
        expected = {82.0: "child", 81.99: "child", 82.01: "unknown", 78.01: "child", 85.0: "unknown",
                    87.99: "unknown", 88.0: "adult", 100.0: "adult", 40.0: "child",
                    150.0: "adult", 39.99: "unknown", 150.01: "unknown"}
        for h, age in expected.items():
            result = classify(geom(h, 500.0), cal, [])
            self.assertEqual(result["age"], age, h)
            self.assertEqual((result["method"], result["ratio"], result["reference_px"]),
                             ("camera_calibration", round(h / 100, 4), 100.0))
        self.assertIn("ratio_between_thresholds", classify(geom(85.0, 500.0), cal, [])["reasons"])
        self.assertIn("implausible_ratio", classify(geom(39.0, 500.0), cal, [])["reasons"])

    def test_calibration_uses_the_linear_plane(self):
        cal = {"status": "ok", "coef": [0.35, 0.01, -200.0], "foot_y_range": [900, 1400], "foot_x_range": [0, 2000]}
        # h_pred = 0.35*1000 + 0.01*1000 - 200 = 160; child <= 0.82 (131.2 px), adult >= 0.88 (140.8 px).
        for h, age, ratio in ((124.8, "child", .78), (131.2, "child", .82), (132.0, "unknown", .825),
                              (140.8, "adult", .88)):
            result = classify(geom(h, 1000.0, 1000.0), cal, [])
            self.assertEqual((result["age"], result["ratio"], result["reference_px"]), (age, ratio, 160.0), h)

    def test_incomplete_or_missing_person_is_unknown(self):
        cal = calibration()
        for g in (geom(50.0, 500.0, complete=False), geom(None, 500.0), geom(50.0, None),
                  geom(-1.0, 500.0), None, {}, "person"):
            result = classify(g, cal, [geom(100.0, 500.0, pid="q")])
            self.assertEqual((result["age"], result["method"], result["ratio"]), ("unknown", "none", None), g)
            self.assertEqual(result["reasons"], ["incomplete_person"])

    def test_extrapolation_beyond_calibrated_range_falls_back(self):
        cal = calibration()  # h_pred 100 -> margin 50 around [400, 600]
        self.assertEqual(classify(geom(70.0, 350.0), cal, [])["method"], "camera_calibration")
        self.assertEqual(classify(geom(70.0, 650.0), cal, [])["method"], "camera_calibration")
        result = classify(geom(70.0, 349.0), cal, [geom(100.0, 349.0, pid="q")])
        self.assertEqual((result["age"], result["method"]), ("child", "relative_height"))
        self.assertIn("outside_calibrated_range", result["reasons"])
        lateral = {"status": "ok", "coef": [0.0, 0.1, 50.0], "foot_y_range": [0, 1000], "foot_x_range": [100, 500]}
        self.assertIn("outside_calibrated_range", classify(geom(100.0, 500.0, 600.0), lateral, [])["reasons"])
        self.assertEqual(classify(geom(100.0, 500.0, 500.0), lateral, [])["method"], "camera_calibration")
        for bounds in (None, ["a", "b"], [1.0], [None, 5.0]):  # unusable bounds: no range guard
            cal = dict(calibration(), foot_y_range=bounds)
            self.assertEqual(classify(geom(70.0, 5000.0), cal, [])["method"], "camera_calibration", bounds)

    def test_unusable_calibrations_fall_back_to_relative_height(self):
        peers = [geom(100.0, 500.0, pid="q")]
        cases = [(None, "no_calibration"), ({}, "no_calibration"),
                 (calibration(status="insufficient"), "calibration_insufficient"),
                 (calibration(status="unreliable"), "calibration_unreliable"),
                 (calibration(coef=[None, 0, 1]), "invalid_calibration"),
                 ({"status": "ok", "coef": [0, 0]}, "invalid_calibration"),
                 (calibration(coef=[1.0, 0.0, -1000.0]), "nonpositive_calibrated_height")]
        for cal, reason in cases:
            result = classify(geom(70.0, 500.0), cal, peers)
            self.assertEqual((result["age"], result["method"], result["ratio"]), ("child", "relative_height", .7), reason)
            self.assertIn(reason, result["reasons"])

    def test_relative_thresholds_are_inclusive(self):
        peers = [geom(100.0, 500.0, pid="q")]
        expected = {75.0: "child", 74.99: "child", 75.01: "unknown", 80.0: "unknown", 89.99: "unknown",
                    90.0: "unknown", 100.0: "unknown", 160.0: "unknown", 30.0: "child", 29.99: "unknown"}
        for h, age in expected.items():
            result = classify(geom(h, 500.0), None, peers)
            self.assertEqual((result["age"], result["method"]), (age, "relative_height"), h)
            self.assertEqual((result["ratio"], result["reference_px"]), (round(h / 100, 4), 100.0))
        self.assertIn("implausible_ratio", classify(geom(20.0, 500.0), None, peers)["reasons"])

    def test_relative_uses_tallest_eligible_peer(self):
        peers = [geom(80.0, 500.0, pid="a"), geom(100.0, 510.0, pid="b"), geom(300.0, 700.0, pid="far")]
        result = classify(geom(78.0, 500.0), None, peers)
        self.assertEqual((result["age"], result["ratio"], result["reference_px"]), ("unknown", .78, 100.0))

    def test_peer_foot_window_is_inclusive(self):
        window = POLICY["peer_foot_window_per_image_height"] * 1000  # 80 px
        for dy in (window, -window):
            result = classify(geom(70.0, 500.0), None, [geom(100.0, 500.0 + dy, pid="q")])
            self.assertEqual(result["age"], "child", dy)
        result = classify(geom(70.0, 500.0), None, [geom(100.0, 580.5, pid="q")])
        self.assertEqual((result["age"], result["method"], result["ratio"]), ("unknown", "none", None))
        self.assertIn("no_eligible_peers", result["reasons"])

    def test_ineligible_peers_are_ignored(self):
        me = geom(70.0, 500.0, pid="me")
        peers = [me, dict(me), geom(100.0, 500.0, pid="other_image", image="img2"),
                 geom(100.0, 500.0, pid="incomplete", complete=False), geom(None, 500.0, pid="noh"),
                 geom(100.0, None, pid="nofoot"), geom(0.0, 500.0, pid="zero"), None, "peer"]
        result = classify(me, calibration(status="insufficient"), peers)
        self.assertEqual((result["age"], result["method"]), ("unknown", "none"))
        self.assertEqual(result["reasons"], ["calibration_insufficient", "no_eligible_peers"])
        self.assertEqual(classify(me, None, None)["reasons"], ["no_calibration", "no_eligible_peers"])

    def test_missing_image_height_disables_relative_fallback(self):
        result = classify(geom(70.0, 500.0, image_height=None), None, [geom(100.0, 500.0, pid="q")])
        self.assertEqual((result["age"], result["method"]), ("unknown", "none"))
        self.assertIn("no_image_height_for_peers", result["reasons"])

    def test_outputs_are_plain_json(self):
        for result in (classify(geom(78.0, 500.0), calibration(), []),
                       classify(geom(np.float32(70.0), np.float64(500.0)), None, [geom(np.float64(100.0), 500.0, pid="q")])):
            self.assertTrue(plain(result), result)
            json.dumps(result)

    def test_end_to_end_adult_with_child_in_same_image(self):
        adult = person_geometry(person())
        child_kps = [[130 + (x - 130) * .6, 390 - (390 - y) * .6, c] for x, y, c in skeleton()]
        child = person_geometry(person(child_kps, box=(110, 210, 150, 395), person_id="p2"))
        self.assertTrue(adult["complete"] and child["complete"], (adult["reasons"], child["reasons"]))
        self.assertTrue(math.isclose(child["height_px"] / adult["height_px"], .6, abs_tol=.01))
        both = [adult, child]
        self.assertEqual(classify(child, None, both)["age"], "child")
        self.assertEqual(classify(adult, None, both)["age"], "unknown")
        self.assertEqual(classify(adult, None, [adult])["method"], "none")


def strict_json(value):
    """Serialise rejecting NaN / Infinity (strict JSON)."""
    return json.dumps(value, allow_nan=False)


# Joint heights above the floor as fractions of stature (Drillis & Contini-style
# segment proportions); the child row approximates a 2-4 year old.
ADULT_BODY = {"eye": .936, "nose": .92, "ear": .93, "shoulder": .818, "elbow": .63, "wrist": .485,
              "hip": .530, "knee": .285, "ankle": .039}
CHILD_BODY = {"eye": .89, "nose": .87, "ear": .88, "shoulder": .76, "elbow": .60, "wrist": .47,
              "hip": .48, "knee": .26, "ankle": .035}
LAYOUT = [(0, "nose"), (.02, "eye"), (-.02, "eye"), (.04, "ear"), (-.04, "ear"), (.10, "shoulder"),
          (-.10, "shoulder"), (.10, "elbow"), (-.10, "elbow"), (.10, "wrist"), (-.10, "wrist"),
          (.06, "hip"), (-.06, "hip"), (.06, "knee"), (-.06, "knee"), (.06, "ankle"), (-.06, "ankle")]


def anatomical(foot_x, sole_y, stature, kind="adult", **extra):
    """Person input built from body proportions; "seated" = adult on a bench 0.25 * stature high."""
    f, crown = dict(CHILD_BODY if kind == "child" else ADULT_BODY), 1.0
    if kind == "seated":  # hip joint 0.30 * stature above the floor, thighs level, shins vertical
        drop = f["hip"] - .30
        for key in ("eye", "nose", "ear", "shoulder", "elbow", "wrist", "hip"):
            f[key] -= drop
        f["knee"], crown = .30, 1.0 - drop
    kps = [[foot_x + dx * stature, sole_y - f[key] * stature, .9] for dx, key in LAYOUT]
    box = [foot_x - .15 * stature, sole_y - crown * stature, foot_x + .15 * stature, sole_y]
    return {"person_id": "p", "image_id": "img", "camera_id": "cam", "box": box, "keypoints": kps,
            "image_width": 2000, "image_height": 1500, "strip_bottom": 0, **extra}


class AnatomicalPipelineTests(unittest.TestCase):
    def test_leg_gate_rejects_bench_sitters_but_keeps_toddlers(self):
        for kind, complete in (("adult", True), ("child", True), ("seated", False)):
            g = person_geometry(anatomical(1000.0, 1200.0, 400.0, kind))
            self.assertEqual(g["complete"], complete, (kind, g["reasons"]))
        # a bench sitter measures ~0.76 of an adult's standing height: never a standing observation
        self.assertEqual(person_geometry(anatomical(1000.0, 1200.0, 400.0, "seated"))["reasons"], ["legs_not_extended"])

    def test_camera_of_adults_toddlers_and_bench_sitters(self):
        rng = np.random.RandomState(21)
        people, kinds = [], []
        for i in range(90):
            kind = "child" if i % 4 == 0 else "seated" if i % 10 == 1 else "adult"
            sole = float(rng.uniform(700, 1400))
            stature = (0.4 * sole - 150) * (1 + .03 * rng.randn()) * (float(rng.uniform(.5, .68)) if kind == "child" else 1)
            people.append(anatomical(float(rng.uniform(300, 1700)), sole, stature, kind,
                                     person_id="p%d" % i, image_id="img%d" % i))
            kinds.append(kind)
        geoms = [person_geometry(p) for p in people]
        for g, kind in zip(geoms, kinds):
            self.assertEqual(g["complete"], kind != "seated", (kind, g["reasons"]))
        cal = fit_camera(geoms)
        self.assertEqual(cal["status"], "ok", cal)
        labels = [classify(g, cal, [])["age"] for g in geoms]
        for kind, expected in (("adult", "adult"), ("child", "child"), ("seated", "unknown")):
            hits = [label == expected for label, truth in zip(labels, kinds) if truth == kind]
            self.assertGreaterEqual(sum(hits) / len(hits), 1.0 if kind == "seated" else .9, kind)
        strict_json([geoms, cal])

    def test_same_image_relative_height_with_real_proportions(self):
        adult = person_geometry(anatomical(800.0, 1200.0, 420.0, person_id="a"))
        toddler = person_geometry(anatomical(1000.0, 1210.0, 240.0, "child", person_id="t"))
        sitter = person_geometry(anatomical(1300.0, 1200.0, 420.0, "seated", person_id="s"))
        people = [adult, toddler, sitter]
        self.assertEqual(classify(toddler, None, people)["age"], "child")
        self.assertEqual(classify(adult, None, people)["age"], "unknown")
        self.assertEqual(classify(sitter, None, people)["reasons"], ["incomplete_person"])
        self.assertEqual(classify(adult, None, [sitter])["reasons"], ["no_calibration", "no_eligible_peers"])


class PersonGeometryAdversarialTests(unittest.TestCase):
    def test_confident_keypoints_far_outside_the_box_are_not_used(self):
        slack = POLICY["keypoint_box_tolerance"] * 300  # box (100, 100, 160, 400): 75 px
        for index, x, y in ((15, 137.0, 400 + slack + .01), (16, 160 + slack + .01, 390.0),
                            (0, 130.0, 100 - slack - .01), (5, -1e200, 170.0), (15, 1e200, 390.0)):
            kps = skeleton()
            kps[index][:2] = [x, y]
            g = person_geometry(person(kps))
            self.assertFalse(g["complete"], index)
            self.assertIn("keypoints_outside_box", g["reasons"])
            self.assertLess(abs(g["foot_x"]), 1000)
            strict_json(g)
        kps = skeleton()
        kps[15][:2] = [137.0, 900.0]  # one stray ankle: the other still defines the foot
        g = person_geometry(person(kps))
        self.assertEqual((g["foot_x"], g["foot_y"], g["foot_source"]), (123.0, 390.0, "ankles"))
        kps[16][:2] = [123.0, 900.0]  # both stray: box bottom
        g = person_geometry(person(kps))
        self.assertEqual((g["foot_y"], g["foot_source"], g["height_px"]), (400.0, "box_bottom", 297.5))
        self.assertEqual(g["reasons"], ["keypoints_outside_box", "no_confident_ankle"])

    def test_keypoint_tolerance_zone_is_inclusive_and_ignores_unconfident_strays(self):
        slack = POLICY["keypoint_box_tolerance"] * 300
        kps = skeleton()
        kps[15][1] = kps[16][1] = 400 + slack
        g = person_geometry(person(kps))
        self.assertTrue(g["complete"], g["reasons"])
        self.assertEqual(g["foot_y"], 475.0)
        kps = skeleton()
        kps[15] = [5000.0, -5000.0, POLICY["keypoint_confidence"] - .001]
        self.assertTrue(person_geometry(person(kps))["complete"])

    def test_overflowing_arithmetic_never_reaches_the_output(self):
        for box in ((0, -1e308, 10, 1e308), (-1e308, 0, 1e308, 10)):  # width/height overflow
            self.assertEqual(person_geometry(person(box=box))["reasons"], ["invalid_box"], box)
        kps = skeleton()  # finite keypoints whose shoulder-eye distance overflows
        for i in (1, 2):
            kps[i][1] = -1.2e308
        for i in (5, 6):
            kps[i][1] = 1.2e308
        g = person_geometry(person(kps, box=(100, -8e307, 160, 8e307)))
        self.assertEqual(g["reasons"], ["nonfinite_geometry"])
        self.assertEqual((g["height_px"], g["foot_y"], g["complete"]), (None, None, False))
        strict_json(g)
        g = person_geometry(person(None, box=(1.5e308, 100, 1.7e308, 400)))
        self.assertIn("touches_right_border", g["reasons"])
        strict_json(g)

    def test_top_data_strip_counts_as_top_border(self):
        touching = person(box=(100, 42, 160, 400), strip_top=40)
        self.assertEqual(person_geometry(touching)["reasons"], ["touches_top_border"])
        self.assertTrue(person_geometry(person(box=(100, 42.5, 160, 400), strip_top=40))["complete"])
        for value in (None, -40, "40", float("inf")):
            self.assertTrue(person_geometry(person(box=(100, 42, 160, 400), strip_top=value))["complete"], value)

    def test_identifiers_come_out_json_safe(self):
        g = person_geometry(person(person_id=np.int64(3), image_id=np.str_("a"), camera_id=["c"]))
        self.assertEqual((g["person_id"], g["image_id"], g["camera_id"]), (3, "a", "['c']"))
        self.assertIs(type(g["person_id"]), int)
        self.assertEqual(person_geometry(person(person_id=float("nan")))["person_id"], "nan")
        strict_json(g)

    def test_only_real_sequences_are_boxes_or_keypoints(self):
        for box in ({100, 101, 160, 400}, b"dd\xa0\xff", {"x1": 100}, "abcd", np.array(5.0),
                    np.zeros((2, 2)), [[100, 100, 160, 400]], iter([100, 100, 160, 400])):
            self.assertEqual(person_geometry(person(box=box))["reasons"], ["invalid_box"], repr(box))
        for box in ((100, 100, 160, 400), np.array([100, 100, 160, 400], dtype=np.int32),
                    np.array([100, 100, 160, 400], dtype=np.float32)):
            self.assertTrue(person_geometry(person(box=box))["complete"], repr(box))
        good = [tuple(k) for k in skeleton()]
        self.assertTrue(person_geometry(person(tuple(good)))["complete"])
        self.assertTrue(person_geometry(person(np.array(skeleton(), dtype=np.float16)))["complete"])
        for bad in ([], {}, np.array(0.0), np.zeros((17, 2)), ["abc"] * 17, [np.array(1.0)] * 17, iter(good)):
            self.assertEqual(person_geometry(person(bad))["reasons"], ["invalid_keypoints"], repr(bad)[:40])

    def test_booleans_and_numpy_bools(self):
        self.assertEqual(person_geometry(person(box=[True, 100, 160, 400]))["reasons"], ["invalid_box"])
        kps = skeleton()
        kps[15][2] = True  # a bool is not a confidence
        self.assertEqual(person_geometry(person(kps))["reasons"], ["invalid_keypoints"])
        g = person_geometry(person())
        self.assertIs(g["complete"], True)
        self.assertEqual(person_geometry(person(image_width=np.bool_(True)))["reasons"], ["invalid_image_dimensions"])


class FitCameraAdversarialTests(unittest.TestCase):
    def test_absurd_rows_neither_warn_nor_move_the_fit(self):
        geoms, _ = synthetic_camera(1)
        clean = fit_camera(geoms)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for field, value in (("height_px", 1e200), ("foot_y", 1e200), ("height_px", 1e-300),
                                 ("foot_y", 1e6), ("foot_x", 1e200), ("foot_y", -1e300)):
                cal = fit_camera(geoms + [dict(geoms[0], person_id="bad", **{field: value})])
                strict_json(cal)
                self.assertIsNotNone(cal["coef"], field)
                self.assertAlmostEqual(cal["coef"][0], clean["coef"][0], delta=.01, msg=field)
                if field != "foot_x" and value > 0:
                    self.assertEqual(cal["status"], "ok", field)
        # an absurd lateral foot widens the observed foot box to where the plane is <= 0
        self.assertEqual(fit_camera(geoms + [dict(geoms[0], foot_x=1e200)])["reasons"],
                         ["nonpositive_prediction_in_range"])

    def test_extreme_magnitudes_are_safe(self):
        rng = np.random.RandomState(12)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for _ in range(20):
                rows = [geom(float(10 ** rng.uniform(-300, 300)), float(rng.choice([-1, 1]) * 10 ** rng.uniform(-300, 300)),
                             float(10 ** rng.uniform(-300, 300)), pid=str(i)) for i in range(25)]
                cal = fit_camera(rows)
                self.assertIn(cal["status"], ("ok", "unreliable"))
                self.assertTrue(plain(cal))
                strict_json(cal)
                strict_json(classify(rows[0], cal, rows))

    def test_bounded_memory_and_time_on_a_huge_camera(self):
        geoms, _ = synthetic_camera(2, n=30000)
        tracemalloc.start()
        try:
            started = time.perf_counter()
            cal = fit_camera(geoms)
            elapsed = time.perf_counter() - started
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertEqual(cal["status"], "ok")
        self.assertLess(peak, 150e6)  # ~40 MB; scoring all candidates on all rows at once took ~400 MB
        self.assertLess(elapsed, 5.0)
        self.assertLessEqual(mean_prediction_error(cal, geoms), .02)

    def test_any_iterable_order_and_duplicates(self):
        geoms, _ = synthetic_camera(3, n=60)
        reference = fit_camera(geoms)
        for variant in (tuple(geoms), iter(geoms), (g for g in geoms), np.array(geoms, dtype=object), geoms[::-1]):
            self.assertEqual(fit_camera(variant), reference)
        doubled = geoms + copy.deepcopy(geoms)
        random.Random(5).shuffle(doubled)
        self.assertEqual(fit_camera(doubled), fit_camera(sorted(doubled, key=lambda g: g["person_id"])))
        for junk in (5, "geoms", {"a": 1}, float("nan"), np.float64(3)):
            self.assertEqual(fit_camera(junk)["status"], "insufficient", junk)

    def test_numpy_bool_complete_counts_but_truthy_junk_does_not(self):
        geoms, _ = synthetic_camera(0, n=30, children=0, sitting=0)
        as_numpy = [dict(g, complete=np.bool_(True)) for g in geoms]
        self.assertEqual(fit_camera(as_numpy), fit_camera(geoms))
        for junk in (1, "yes", np.bool_(False), [True]):
            self.assertEqual(fit_camera([dict(g, complete=junk) for g in geoms])["n"], 0, junk)

    def test_min_people_argument(self):
        geoms, _ = synthetic_camera(0, n=6, children=0, sitting=0)
        for value, status in ((None, "ok"), (0, "ok"), (-3, "ok"), (6.0, "ok"), (6.5, "insufficient"), (7, "insufficient")):
            self.assertEqual(fit_camera(geoms, min_people=value)["status"], status, value)

    def test_identical_people_give_an_exact_constant_model(self):
        cal = fit_camera([geom(150.0, 1000.0, 20.0, "p%d" % i) for i in range(20)])
        self.assertEqual((cal["status"], cal["model"], cal["coef"], cal["residual_mad"], cal["inliers"]),
                         ("ok", "constant", [0.0, 0.0, 150.0], 0.0, 20))

    def test_inlier_fraction_boundary(self):
        wild = [30.0, 45.0, 160.0, 200.0, 250.0, 300.0, 40.0, 170.0, 230.0, 35.0, 260.0]
        for exact, status in ((10, "ok"), (9, "unreliable")):
            rows = ([geom(100.0, 1000.0, float(i), "a%d" % i) for i in range(exact)]
                    + [geom(w, 1000.0, 0.0, "w%d" % i) for i, w in enumerate(wild[:20 - exact])])
            cal = fit_camera(rows)
            self.assertEqual(cal["status"], status, exact)
        self.assertIn("low_inlier_fraction", cal["reasons"])
        half = fit_camera([geom(100.0, 1000.0, float(i), "a%d" % i) for i in range(10)]
                          + [geom(w, 1000.0, 0.0, "w%d" % i) for i, w in enumerate(wild[:10])])
        self.assertEqual((half["inliers"], half["inlier_fraction"], half["coef"][2]), (10, .5, 100.0))

    def test_residual_mad_limits_are_inclusive(self):
        limit = POLICY["max_residual_mad_constant"]
        for spread, status in ((limit, "ok"), (limit + .0001, "unreliable")):
            rows = [geom(100 * (1 + (spread if i % 2 else -spread)), 1000.0, 10.0 * i, "p%d" % i) for i in range(20)]
            cal = fit_camera(rows)
            self.assertEqual((cal["status"], cal["model"], cal["residual_mad"]), (status, "constant", round(spread, 4)))
        geoms, _ = synthetic_camera(4)
        mad = fit_camera(geoms)["residual_mad"]
        with mock.patch.dict(POLICY, max_residual_mad=mad):
            self.assertEqual(fit_camera(geoms)["status"], "ok")
        with mock.patch.dict(POLICY, max_residual_mad=mad - .0001):
            self.assertEqual(fit_camera(geoms)["reasons"], ["residual_mad_above_limit"])


class ClassifyAdversarialTests(unittest.TestCase):
    def test_relative_ratio_has_a_plausibility_ceiling(self):
        peers = [geom(100.0, 500.0, pid="q")]
        for h, age in ((POLICY["relative_ratio_max"] * 100, "unknown"), (POLICY["relative_ratio_max"] * 100 + .01, "unknown")):
            self.assertEqual(classify(geom(h, 500.0), None, peers)["age"], age, h)
        self.assertIn("implausible_ratio", classify(geom(300.01, 500.0), None, peers)["reasons"])
        result = classify(geom(300.0, 500.0), None, [geom(1e-310, 500.0, pid="q")])  # ratio overflows
        self.assertEqual((result["age"], result["method"], result["ratio"]), ("unknown", "relative_height", None))
        self.assertIn("implausible_ratio", result["reasons"])
        strict_json(result)

    def test_similar_height_peers_cannot_establish_adult_age(self):
        # Two equally sized children must not label one another as adult;
        # more unlabelled peers do not provide an adult anchor either.
        for count in (1, 2, 20):
            peers = [geom(100.0, 500.0, pid=f"peer_{i}") for i in range(count)]
            result = classify(geom(100.0, 500.0), None, peers)
            self.assertEqual((result["age"], result["ratio"]), ("unknown", 1.0))
            self.assertIn("no_verified_adult_reference", result["reasons"])
            self.assertEqual(classify(geom(70.0, 500.0), None, peers)["age"], "child")

    def test_exotic_or_overflowing_calibrations(self):
        peers = [geom(100.0, 500.0, pid="q")]
        for cal, reason in (({"status": "ok", "coef": [1e308, 0.0, 1e308]}, "invalid_calibration"),
                            ({"status": "ok", "coef": "abc"}, "invalid_calibration"),
                            ({"status": "ok", "coef": [0.0, 0.0, float("nan")]}, "invalid_calibration"),
                            ({"status": np.array(["ok", "ok"])}, "no_calibration"),
                            ({"status": 5}, "no_calibration"), ("ok", "no_calibration")):
            result = classify(geom(70.0, 500.0), cal, peers)
            self.assertEqual((result["age"], result["method"]), ("child", "relative_height"), reason)
            self.assertIn(reason, result["reasons"])
            strict_json(result)
        numpy_cal = {"status": np.str_("ok"), "coef": np.array([0.0, 0.0, 100.0]),
                     "foot_y_range": np.array([400.0, 600.0]), "foot_x_range": (0, 1000)}
        result = classify(geom(70.0, 500.0), numpy_cal, [])
        self.assertEqual((result["age"], result["method"], result["ratio"]), ("child", "camera_calibration", .7))
        self.assertTrue(plain(result))
        self.assertEqual(classify(geom(70.0, 5000.0), numpy_cal, [])["reasons"],
                         ["outside_calibrated_range", "no_eligible_peers"])

    def test_peer_containers_and_order(self):
        me = geom(70.0, 500.0, pid="me")
        peers = [geom(100.0, 500.0, pid="a"), geom(90.0, 510.0, pid="b"), geom(300.0, 900.0, pid="far")]
        reference = classify(me, None, peers)
        self.assertEqual((reference["age"], reference["ratio"]), ("child", .7))
        for variant in (tuple(peers), iter(peers), np.array(peers, dtype=object), peers[::-1]):
            self.assertEqual(classify(me, None, variant), reference)
        for junk in (5, "peers", {"a": 1}):
            self.assertEqual(classify(me, None, junk)["reasons"], ["no_calibration", "no_eligible_peers"])
        self.assertEqual(classify(me, None, [dict(peers[0], complete=np.bool_(True))])["age"], "child")
        self.assertEqual(classify(dict(me, complete=np.bool_(True)), None, peers)["age"], "child")
        self.assertEqual(classify(me, None, [dict(peers[0], complete=1)])["method"], "none")
        weird = classify(dict(me, image_id=np.array([1, 2])), None, [dict(peers[0], image_id=np.array([1, 2]))])
        self.assertEqual(weird["age"], "child")  # array ids compare by value, never raise

    def test_self_is_never_a_peer(self):
        me = geom(100.0, 500.0, pid=np.int64(7))
        self.assertEqual(classify(me, None, [me, dict(me), dict(me, person_id=7)])["method"], "none")

    def test_inputs_are_not_mutated(self):
        me, peers, cal = geom(70.0, 500.0, pid="me"), [geom(100.0, 500.0, pid="q")], calibration()
        geoms, _ = synthetic_camera(5, n=40)
        before = copy.deepcopy((me, peers, cal, geoms))
        classify(me, cal, peers)
        classify(me, None, peers)
        fit_camera(geoms)
        self.assertEqual((me, peers, cal, geoms), before)


class FuzzTests(unittest.TestCase):
    ATOMS = [None, float("nan"), float("inf"), -float("inf"), 0, -1, 1e308, -1e308, 5e-324, True, np.bool_(True),
             "x", b"ab", np.float32(3), np.int64(5), [], {}, set(), (1, 2), np.array(3.0), np.zeros((2, 2)),
             7.5, 400.0, "ok"]

    def test_malformed_inputs_never_raise_and_outputs_stay_strict_json(self):
        rng = random.Random(1234)

        def junk(depth=0):
            roll = rng.random()
            if depth > 2 or roll < .6:
                return rng.choice(self.ATOMS)
            if roll < .8:
                return [junk(depth + 1) for _ in range(rng.choice([0, 2, 3, 4, 17]))]
            return {rng.choice(["box", "keypoints", "complete", "height_px", "foot_y", "status", "coef"]): junk(depth + 1)}

        def mutate(record):
            record = dict(record)
            for _ in range(rng.randint(1, 3)):
                record[rng.choice(list(record) + ["extra"])] = junk()
            return record

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for _ in range(1500):
                p = mutate(person()) if rng.random() < .7 else junk()
                if isinstance(p, dict) and rng.random() < .4:
                    kps = skeleton()
                    for _ in range(rng.randint(1, 5)):
                        kps[rng.randrange(17)][rng.randrange(3)] = rng.choice(self.ATOMS)
                    p["keypoints"] = kps
                g = person_geometry(p)
                rows = [mutate(geom(100.0 + i, 500.0 + 3 * i, pid="q%d" % i)) if rng.random() < .5
                        else geom(100.0 + i, 500.0 + 3 * i, pid="q%d" % i) for i in range(rng.choice([0, 3, 6, 25]))]
                cal = fit_camera(rows if rng.random() < .8 else junk(), min_people=rng.choice([20, 5, None, "x"]))
                use = cal if rng.random() < .6 else mutate(calibration())
                target = g if rng.random() < .5 else (rows[0] if rows else junk())
                c = classify(target, use, rows if rng.random() < .8 else junk())
                for out in (g, cal, c):
                    self.assertTrue(plain(out), out)
                    strict_json(out)
                self.assertIn(c["age"], ("adult", "child", "unknown"))
                self.assertIn(c["method"], ("camera_calibration", "relative_height", "none"))
                self.assertIn(cal["status"], ("ok", "insufficient", "unreliable"))
                self.assertIs(type(g["complete"]), bool)


if __name__ == "__main__":
    unittest.main()
