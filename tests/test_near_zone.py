"""Near-zone checks (no models, no photos): ``expected_height`` gating and ``people_near``.

Per person the reference height is the calibrated standing-adult height at the
foot position (``age_geometry.expected_height``; only for an ``ok``
calibration and a foot inside the calibrated range plus the extrapolation
margin), else the largest own height among the people at the same depth or
farther: foot row at most 0.02 x image height below the person's (the person
included). The rule is monotonic in depth: anyone closer to the camera than a
near person is near too. A person is near when that reference is at least
``near_fraction`` x image height (explicit test setting 0.07; default 0 counts everyone). Images
report ``people_near``, events ``people_near_max_frame`` and
``people_near_unique`` (tracks with any near observation), camera days sum
both, and every calibration group reports a ``near_zone`` diagnostic.
"""
from __future__ import annotations

import copy
import json
import math
import unittest
from unittest.mock import patch

try:
    from . import test_postprocess as helpers
except ImportError:  # unittest discover -s tests imports test modules top-level
    import test_postprocess as helpers

from trailcam.age_geometry import POLICY, classify, expected_height, person_geometry
from trailcam.export import make_event_row, make_row, summary_rows
from trailcam.postprocess import DEFAULTS

experimental_postprocess = helpers.experimental_postprocess

image, item, walker, at, accepted = helpers.image, helpers.item, helpers.walker, helpers.at, helpers.accepted
WIDTH, HEIGHT = helpers.WIDTH, helpers.HEIGHT  # 2000 x 1200: depth window 24 px, default limit ~84 px
CAL = {"status": "ok", "coef": [0.5, 0.0, 10.0], "foot_y_range": [200.0, 800.0], "foot_x_range": [100.0, 1900.0]}


def still(box, colour=0):
    """A person without pose keypoints: own height = box height, foot = box bottom."""
    return walker(box, colour=colour, pose=False)


def run(people, settings=None, path="camA/IMG_0001.JPG"):
    frame = item(path, image(people, at(0)))
    post = experimental_postprocess([frame], settings)
    return frame["result"], post


def by_left(result):
    """combined person fields keyed by the box's left edge."""
    return {p["xyxy"][0]: p["combined"] for p in accepted(result)}


class ExpectedHeightTests(unittest.TestCase):
    def test_value_at_the_foot_row(self):
        self.assertEqual(expected_height({"foot_y": 400.0, "foot_x": 1000.0}, CAL), 210.0)
        self.assertEqual(expected_height({"foot_y": 800, "foot_x": 100}, CAL), 410.0)  # ints are fine

    def test_only_ok_calibrations_are_used(self):
        geom = {"foot_y": 400.0, "foot_x": 1000.0}
        for status in ("unreliable", "insufficient", None, "OK", "ok ", 1, True):
            with self.subTest(status=status):
                self.assertIsNone(expected_height(geom, {**CAL, "status": status}))
        self.assertIsNone(expected_height(geom, {k: v for k, v in CAL.items() if k != "status"}))
        for calibration in (None, [], "ok", 0):
            with self.subTest(calibration=calibration):
                self.assertIsNone(expected_height(geom, calibration))

    def test_geometry_must_carry_a_finite_foot_row(self):
        for geom in (None, [], "geometry", {}, {"foot_y": None}, {"foot_y": True}, {"foot_y": math.nan},
                     {"foot_y": math.inf}, {"foot_y": "400"}, {"foot_x": 1000.0}):
            with self.subTest(geom=geom):
                self.assertIsNone(expected_height(geom, CAL))

    def test_extrapolation_margin_is_half_the_predicted_height_inclusive(self):
        self.assertEqual(POLICY["calibration_extrapolation_per_height"], 0.5)
        cal = {**CAL, "coef": [0.5, 0.0, 0.0], "foot_y_range": [200.0, 900.0]}
        # Lower edge: 160 - 0.5 * (0.5 * 160) == 200 - 40 == 160.
        self.assertEqual(expected_height({"foot_y": 160.0}, cal), 80.0)
        self.assertIsNone(expected_height({"foot_y": 159.99}, cal))
        # Upper edge: 900 + 0.5 * (0.5 * 1200) == 1200.
        self.assertEqual(expected_height({"foot_y": 1200.0}, cal), 600.0)
        self.assertIsNone(expected_height({"foot_y": 1200.01}, cal))
        self.assertEqual(expected_height({"foot_y": 500.0}, cal), 250.0)

    def test_foot_x_only_matters_with_a_lateral_term(self):
        # b == 0: foot_x may be missing or far outside the observed x range.
        self.assertEqual(expected_height({"foot_y": 400.0}, CAL), 210.0)
        self.assertEqual(expected_height({"foot_y": 400.0, "foot_x": 99999.0}, CAL), 210.0)
        lateral = {**CAL, "coef": [0.5, 0.1, 0.0]}
        self.assertIsNone(expected_height({"foot_y": 400.0}, lateral))              # missing_foot_x
        self.assertIsNone(expected_height({"foot_y": 400.0, "foot_x": None}, lateral))
        self.assertAlmostEqual(expected_height({"foot_y": 400.0, "foot_x": 1000.0}, lateral), 300.0)
        self.assertAlmostEqual(expected_height({"foot_y": 400.0, "foot_x": 2000.0}, lateral), 400.0)  # within margin
        self.assertIsNone(expected_height({"foot_y": 400.0, "foot_x": 5000.0}, lateral))  # 5000 > 1900 + 350

    def test_invalid_or_nonpositive_calibrations_give_none(self):
        geom = {"foot_y": 400.0, "foot_x": 1000.0}
        for coef in (None, [], [0.5, 10.0], "0.5,0,10", [0.5, 0.0, math.nan], [0.5, math.inf, 0.0],
                     [0.5, 0.0, "10"], [0.5, 0.0, -300.0], [0.5, 0.0, -200.0]):
            with self.subTest(coef=coef):
                self.assertIsNone(expected_height(geom, {**CAL, "coef": coef}))
        overflow = {**CAL, "coef": [1e308, 0.0, 1e308], "foot_y_range": [0.0, 1e12]}
        self.assertIsNone(expected_height({"foot_y": 1e10}, overflow))

    def test_missing_ranges_skip_the_range_check(self):
        bare = {"status": "ok", "coef": [0.5, 0.0, 10.0]}
        self.assertEqual(expected_height({"foot_y": 5000.0}, bare), 2510.0)
        self.assertEqual(expected_height({"foot_y": 5000.0}, {**bare, "foot_y_range": [1.0]}), 2510.0)

    def test_rounded_to_three_decimals(self):
        cal = {**CAL, "coef": [1 / 3, 0.0, 0.0], "foot_y_range": [0.0, 1000.0]}
        self.assertEqual(expected_height({"foot_y": 100.0}, cal), 33.333)

    def test_agrees_with_the_calibrated_age_reference(self):
        box = [900.0, 300.0, 1000.0, 600.0]
        geom = person_geometry({"person_id": "p", "image_id": "i", "camera_id": "c", "box": box,
                                "keypoints": helpers.standing_keypoints(box), "image_width": WIDTH,
                                "image_height": HEIGHT})
        self.assertTrue(geom["complete"], geom["reasons"])
        decision = classify(geom, CAL, [])
        self.assertEqual(decision["method"], "camera_calibration")
        reference = expected_height(geom, CAL)
        self.assertAlmostEqual(reference, 0.5 * geom["foot_y"] + 10.0, places=3)
        self.assertAlmostEqual(decision["reference_px"], reference, places=2)

    def test_incomplete_people_still_get_a_reference(self):
        geom = {"complete": False, "foot_y": 400.0, "foot_x": 1000.0, "height_px": 50.0}
        self.assertEqual(classify(geom, CAL, [])["method"], "none")   # no age from an incomplete person...
        self.assertEqual(expected_height(geom, CAL), 210.0)          # ...but a distance-like reference

    def test_inputs_are_not_mutated(self):
        geom, cal = {"foot_y": 400.0, "foot_x": 1000.0}, copy.deepcopy(CAL)
        before = copy.deepcopy((geom, cal))
        expected_height(geom, cal)
        self.assertEqual((geom, cal), before)


class NearZoneWithoutCalibrationTests(unittest.TestCase):
    """One photo on a camera with too few people for a calibration."""

    def test_default_fraction_and_settings(self):
        self.assertEqual(DEFAULTS["near_fraction"], 0.0)
        _, post = run([])
        self.assertEqual(post["settings"]["near_fraction"], 0.07)
        _, post = run([], {"near_fraction": 0.25})
        self.assertEqual(post["settings"]["near_fraction"], 0.25)
        self.assertEqual(post["calibrations"]["camA|Acme TC-1@2000x1200"]["status"], "insufficient")

    def test_lone_people_are_judged_by_their_own_height(self):
        result, _ = run([still([100, 500, 140, 600]), still([800, 200, 830, 280], colour=3)])
        people = by_left(result)
        self.assertEqual((people[100]["near"], people[100]["near_reference_px"]), (True, 100.0))
        self.assertEqual((people[800]["near"], people[800]["near_reference_px"]), (False, 80.0))
        self.assertEqual({p["near_source"] for p in people.values()}, {"height_at_similar_depth"})
        self.assertEqual((result["combined"]["counts"]["people_total"], result["combined"]["people_near"]), (2, 1))

    def test_measured_crown_to_foot_height_is_used_when_available(self):
        result, _ = run([walker([100, 300, 200, 600])])
        person = accepted(result)[0]
        self.assertTrue(person["geometry"]["complete"])
        self.assertEqual(person["combined"]["near_reference_px"], round(person["geometry"]["height_px"], 1))
        self.assertNotEqual(person["combined"]["near_reference_px"], 300.0)  # not the box height

    def test_short_person_beside_a_tall_one_is_near(self):
        result, _ = run([still([100, 300, 200, 600]),                 # tall, foot row 600
                         still([300, 540, 330, 600], colour=3),       # short, same foot row
                         still([1000, 640, 1030, 700], colour=5)])    # short, 100 px lower (closer)
        people = by_left(result)
        self.assertEqual((people[300]["near"], people[300]["near_reference_px"]), (True, 300.0))
        # Changed: peers are people at the same depth or farther, so a short person
        # closer to the camera than the near tall one is near too (was: alone, 60 px).
        self.assertEqual((people[1000]["near"], people[1000]["near_reference_px"]), (True, 300.0))
        self.assertEqual(people[100]["near_reference_px"], 300.0)   # the tall one keeps its own height
        self.assertEqual(result["combined"]["people_near"], 3)

    def test_near_is_monotonic_in_depth(self):
        tall = still([100, 300, 200, 600])                                 # near by its own height, foot 600
        closer = [still([400, 840, 420, 900], colour=3),                   # 300 px closer, 60 px tall
                  still([700, 564.5, 720, 624.5], colour=5)]               # 24.5 px closer
        farther = [still([1000, 440, 1020, 500], colour=7),                # 100 px farther, 60 px tall
                   still([1300, 515.5, 1320, 575.5], colour=9)]            # 24.5 px farther: outside the window
        result, _ = run([tall, *closer, *farther])
        people = by_left(result)
        for left in (100, 400, 700):
            with self.subTest(left=left):
                self.assertEqual((people[left]["near"], people[left]["near_reference_px"],
                                  people[left]["near_source"]), (True, 300.0, "height_at_similar_depth"))
        for left in (1000, 1300):
            with self.subTest(left=left):
                self.assertEqual((people[left]["near"], people[left]["near_reference_px"]), (False, 60.0))
        self.assertEqual(result["combined"]["people_near"], 3)
        # Monotonic: everyone whose foot row is at or below a near person's is near.
        feet = {p["xyxy"][0]: p["geometry"]["foot_y"] for p in accepted(result)}
        lowest_near = min(feet[x] for x, p in people.items() if p["near"])
        self.assertEqual({x for x in feet if feet[x] >= lowest_near}, {x for x, p in people.items() if p["near"]})

    def test_similar_depth_window_is_two_percent_of_the_image_height_inclusive(self):
        self.assertEqual(0.02 * HEIGHT, 24.0)
        # Changed: the 24 px window now only reaches toward the camera (a peer may be
        # at most 24 px lower); anyone lower still is near anyway (monotonic rule).
        result, _ = run([still([100, 300, 200, 600]),                  # tall, foot 600
                         still([400, 516, 420, 576], colour=3),        # foot 576: the tall one is exactly 24 px lower
                         still([600, 515.5, 620, 575.5], colour=5),    # foot 575.5: the tall one is 24.5 px lower
                         still([800, 564.5, 820, 624.5], colour=7)])   # foot 624.5: closer than the tall one
        people = by_left(result)
        self.assertEqual((people[400]["near"], people[400]["near_reference_px"]), (True, 300.0))
        # Within 0.5 px of the 576 person, but its only peers (576, itself) are short: reference 60.
        self.assertEqual((people[600]["near"], people[600]["near_reference_px"]), (False, 60.0))
        # Was (False, 60) with the symmetric window; closer than a near person is near.
        self.assertEqual((people[800]["near"], people[800]["near_reference_px"]), (True, 300.0))

    def test_limit_is_inclusive(self):
        settings = {"near_fraction": 0.25}                         # 0.25 * 1200 == 300 exactly
        result, _ = run([still([100, 300, 200, 600]), still([600, 100.5, 700, 400], colour=3)], settings)
        people = by_left(result)
        self.assertEqual((people[100]["near"], people[100]["near_reference_px"]), (True, 300.0))
        self.assertEqual((people[600]["near"], people[600]["near_reference_px"]), (False, 299.5))

    def test_zero_or_missing_fraction_counts_everyone(self):
        people = [still([100, 100, 102, 105]), still([400, 500, 404, 510], colour=3), still([900, 300, 1000, 600], colour=5)]
        for fraction in (0, 0.0, None):
            with self.subTest(fraction=fraction):
                result, _ = run(copy.deepcopy(people), {"near_fraction": fraction})
                self.assertEqual(result["combined"]["counts"]["people_total"], 3)
                self.assertEqual(result["combined"]["people_near"], 3)
                self.assertTrue(all(p["near"] for p in by_left(result).values()))

    def test_large_fraction_counts_no_one(self):
        result, _ = run([still([100, 300, 200, 600]), walker([900, 100, 1100, 700], colour=3)], {"near_fraction": 0.9})
        self.assertEqual(result["combined"]["people_near"], 0)
        self.assertFalse(any(p["near"] for p in by_left(result).values()))

    def test_empty_photo_has_zero_near_people(self):
        result, _ = run([])
        self.assertEqual(result["combined"]["people_near"], 0)

    def test_near_zone_leaves_ages_and_totals_unchanged(self):
        people = [walker([300, 300, 400, 900], colour=1), walker([600, 540, 700, 900], colour=3)]
        outcomes = []
        for fraction in (0, 0.5, 0.95):
            result, _ = run(copy.deepcopy(people), {"near_fraction": fraction})
            combined = result["combined"]
            outcomes.append((combined["adults"], combined["children"], combined["counts"]["people_total"]))
        self.assertEqual(outcomes, [(0, 1, 2)] * 3)

    def test_persons_json_reports_near_and_its_source(self):
        result, _ = run([still([100, 500, 140, 600]), still([800, 200, 830, 280], colour=3)])
        row = make_row(result, {"run_id": "r", "image_id": "i", "relative_path": "camA/IMG_0001.JPG"})
        people = sorted(json.loads(row["persons_json"]), key=lambda p: p["box"][0])
        self.assertEqual([(p["near"], p["near_source"]) for p in people],
                         [(True, "height_at_similar_depth"), (False, "height_at_similar_depth")])
        self.assertEqual((row["people_near"], row["people_near_class"], row["people_class"]), (1, "1", "2"))


class CalibratedNearZoneTests(unittest.TestCase):
    """24 adults whose height grows with the foot row calibrate camera ``calib``."""

    @classmethod
    def setUpClass(cls):
        def box(bottom, scale=1.0):
            height = 0.4 * bottom * scale
            return [900, bottom - height, 900 + height / 3, bottom]

        adults = [item(f"calib/IMG_{n:04d}.JPG", image([walker(box(500 + 25 * n), colour=n)], at(1000 * n), n))
                  for n in range(24)]
        cls.child = item("calib/IMG_0100.JPG", image([walker(box(800, 0.6), colour=99)], at(30000), 100))
        cls.sitting = item("calib/IMG_0101.JPG", image([still([300, 700, 380, 800], colour=98)], at(31000), 101))
        cls.far = item("calib/IMG_0102.JPG", image([still([300, 90, 320, 150], colour=97)], at(32000), 102))
        cls.other = item("elsewhere/IMG_0001.JPG", image([walker(box(800, 0.6), colour=99)], at(0), 1))
        # Limit 0.2 * 1200 = 240 px: above the child's own height (~188 px), below the
        # calibrated adult height at its foot row (~315 px).
        cls.post = experimental_postprocess(adults + [cls.child, cls.sitting, cls.far, cls.other], {"near_fraction": 0.2})
        cls.calibration = cls.post["calibrations"]["calib|Acme TC-1@2000x1200"]

    def person(self, frame):
        people = accepted(frame["result"])
        self.assertEqual(len(people), 1)
        return people[0]

    def test_calibration_is_ok(self):
        self.assertEqual(self.calibration["status"], "ok", self.calibration)
        self.assertEqual(self.post["calibrations"]["elsewhere|Acme TC-1@2000x1200"]["status"], "insufficient")

    def test_calibrated_reference_inside_the_range(self):
        person = self.person(self.child)
        combined = person["combined"]
        reference = expected_height(person["geometry"], self.calibration)
        self.assertIsNotNone(reference)
        self.assertEqual(combined["near_source"], "camera_calibration")
        self.assertEqual(combined["near_reference_px"], round(reference, 1))
        self.assertLess(person["geometry"]["height_px"], 0.2 * HEIGHT)   # own height alone would be far
        self.assertGreaterEqual(reference, 0.2 * HEIGHT)
        self.assertTrue(combined["near"])
        self.assertEqual(self.child["result"]["combined"]["people_near"], 1)

    def test_incomplete_person_inside_the_range_uses_the_calibration(self):
        person = self.person(self.sitting)
        self.assertFalse(person["geometry"]["complete"])
        self.assertEqual(person["combined"]["near_source"], "camera_calibration")
        self.assertTrue(person["combined"]["near"])
        self.assertEqual(person["combined"]["near_reference_px"],
                         round(expected_height(person["geometry"], self.calibration), 1))

    def test_outside_the_calibrated_range_falls_back_to_own_height(self):
        person = self.person(self.far)
        self.assertIsNone(expected_height(person["geometry"], self.calibration))
        self.assertEqual((person["combined"]["near_source"], person["combined"]["near_reference_px"],
                          person["combined"]["near"]), ("height_at_similar_depth", 60.0, False))

    def test_same_person_on_an_uncalibrated_camera_is_far(self):
        person = self.person(self.other)
        self.assertEqual(person["combined"]["near_source"], "height_at_similar_depth")
        self.assertEqual(person["combined"]["near_reference_px"], round(person["geometry"]["height_px"], 1))
        self.assertFalse(person["combined"]["near"])
        self.assertEqual(self.other["result"]["combined"]["people_near"], 0)


class InjectedCalibrationTests(unittest.TestCase):
    """``fit_camera`` replaced by a fixed calibration: h = 0.5 * foot_y, feet 200..900."""

    def run_with(self, calibration, people, settings=None):
        with patch("trailcam.age_geometry.fit_camera", return_value=calibration):
            return run(people, settings)

    def test_calibration_beats_similar_depth_peers(self):
        calibration = {"status": "ok", "coef": [0.5, 0.0, 0.0], "foot_y_range": [200.0, 900.0],
                       "foot_x_range": [0.0, 2000.0]}
        # Beside a 300 px tall person, but the camera says an adult at foot row 400 is 200 px.
        result, _ = self.run_with(calibration, [still([100, 300, 200, 400]), still([300, 370, 320, 400], colour=3)],
                                  {"near_fraction": 0.2})  # limit 240
        people = by_left(result)
        self.assertEqual([(people[x]["near"], people[x]["near_reference_px"], people[x]["near_source"]) for x in (100, 300)],
                         [(False, 200.0, "camera_calibration")] * 2)
        self.assertEqual(result["combined"]["people_near"], 0)
        self.assertEqual(result["post"]["camera_calibration_status"], "ok")

    def test_unreliable_calibration_is_ignored(self):
        calibration = {"status": "unreliable", "coef": [0.5, 0.0, 0.0], "foot_y_range": [200.0, 900.0],
                       "foot_x_range": [0.0, 2000.0]}
        result, _ = self.run_with(calibration, [still([100, 300, 200, 400]), still([300, 370, 320, 400], colour=3)],
                                  {"near_fraction": 0.2})
        people = by_left(result)
        self.assertEqual([(people[x]["near"], people[x]["near_reference_px"], people[x]["near_source"]) for x in (100, 300)],
                         [(False, 100.0, "height_at_similar_depth")] * 2)  # peers only: 100 < 240

    def test_mixed_sources_in_one_photo(self):
        calibration = {"status": "ok", "coef": [0.5, 0.0, 0.0], "foot_y_range": [500.0, 900.0],
                       "foot_x_range": [0.0, 2000.0]}
        # Foot 600 is calibrated (reference 300); foot 150 is outside 500 - 0.5 * 75.
        result, _ = self.run_with(calibration, [still([100, 560, 120, 600]), still([800, 20, 900, 150], colour=3)],
                                  {"near_fraction": 0.1})  # limit 120
        people = by_left(result)
        self.assertEqual((people[100]["near_source"], people[100]["near_reference_px"], people[100]["near"]),
                         ("camera_calibration", 300.0, True))
        self.assertEqual((people[800]["near_source"], people[800]["near_reference_px"], people[800]["near"]),
                         ("height_at_similar_depth", 130.0, True))


class EventNearTests(unittest.TestCase):
    """Three frames 5 s apart: A approaches (near only in the last frame), D is near
    throughout, B stays far; a second event later that day has one near person.

    D stands closer to the camera than A (foot row 720 vs 670-690, more than the
    24 px window): with the monotonic near rule a person closer than a near one is
    near, so D in front keeps A judged by its own height. (With D's feet at row
    600, behind A, A would now be near in every frame.)"""

    @staticmethod
    def walk(d_foot=720):
        frames = []
        for n in range(3):
            a = still([500 + 20 * n, 600, 520 + 22 * n, 670 + 10 * n], colour=10)   # 70, 80, 90 px tall
            d = still([1000 + 20 * n, d_foot - 200, 1080 + 20 * n, d_foot], colour=30)  # 200 px, foot 720
            b = still([1500, 300, 1515, 340], colour=50)                             # 40 px
            frames.append(item(f"walk/IMG_{n:04d}.JPG", image([a, d, b], at(5 * n), n)))
        return frames

    @classmethod
    def setUpClass(cls):
        cls.frames = cls.walk()
        cls.later = item("walk/IMG_0100.JPG", image([still([300, 300, 400, 600], colour=70)], at(3600), 100))
        cls.post = experimental_postprocess(cls.frames + [cls.later])
        cls.event = next(e for e in cls.post["events"] if e["image_count"] == 3)
        cls.single = next(e for e in cls.post["events"] if e["image_count"] == 1)

    def test_per_frame_near_counts(self):
        self.assertEqual([f["result"]["combined"]["people_near"] for f in self.frames], [1, 1, 2])
        self.assertEqual([f["result"]["combined"]["counts"]["people_total"] for f in self.frames], [3, 3, 3])

    def test_a_person_behind_a_near_one_does_not_borrow_its_height(self):
        # A (feet 670-690) is farther than D (feet 720) by more than the window: own height.
        references = [{p["xyxy"][1]: p["combined"]["near_reference_px"] for p in accepted(f["result"])}
                      for f in self.frames]
        self.assertEqual([r[600] for r in references], [70.0, 80.0, 90.0])   # A
        self.assertEqual([r[520] for r in references], [200.0] * 3)          # D
        self.assertEqual([r[300] for r in references], [40.0] * 3)           # B

    def test_moving_the_near_person_behind_makes_everyone_in_front_near(self):
        # The same walk with D's feet at row 600 (behind A): A is closer than a near
        # person in every frame and therefore near in every frame.
        frames = self.walk(d_foot=600)
        experimental_postprocess(frames)
        self.assertEqual([f["result"]["combined"]["people_near"] for f in frames], [2, 2, 2])
        a = [next(p for p in accepted(f["result"]) if p["xyxy"][1] == 600) for f in frames]
        self.assertEqual([(p["combined"]["near"], p["combined"]["near_reference_px"]) for p in a], [(True, 200.0)] * 3)

    def test_each_person_is_one_track(self):
        tracks = {}
        for frame in self.frames:
            for person in accepted(frame["result"]):
                tracks.setdefault(person["xyxy"][1], set()).add(person["track_id"])  # top edge differs per person
        self.assertEqual([len(ids) for ids in tracks.values()], [1, 1, 1])
        self.assertEqual(len(set().union(*tracks.values())), 3)

    def test_event_near_maxima_and_unique_tracks(self):
        event = self.event
        self.assertEqual((event["people_unique"], event["people_max_frame"]), (3, 3))
        self.assertEqual((event["people_near_max_frame"], event["people_near_unique"]), (2, 2))
        self.assertEqual((self.single["people_near_max_frame"], self.single["people_near_unique"]), (1, 1))

    def test_event_rows_carry_near_columns_and_classes(self):
        row = make_event_row(self.event, "run_1")
        self.assertEqual((row["people_near_max_frame"], row["people_near_unique"]), (2, 2))
        self.assertEqual((row["people_unique_class"], row["people_near_unique_class"]), ("3–4", "2"))

    def test_camera_days_sum_near_columns(self):
        day, = summary_rows(self.post["events"], "run_1")
        self.assertEqual((day["events"], day["people_unique"], day["people_max_frame"]), (2, 4, 4))
        self.assertEqual((day["people_near_max_frame"], day["people_near_unique"]), (3, 3))

    def test_zero_fraction_makes_near_equal_to_total(self):
        frames = self.walk()
        post = experimental_postprocess(frames, {"near_fraction": 0})
        self.assertEqual([f["result"]["combined"]["people_near"] for f in frames], [3, 3, 3])
        event, = post["events"]
        self.assertEqual((event["people_near_max_frame"], event["people_near_unique"]), (3, 3))


class NearZoneDiagnosticTests(unittest.TestCase):
    """``calibrations[key]["near_zone"]``: per calibration group, people_total and
    people_near summed over its images; ``check_near_fraction`` when at least 10
    people were seen but none was near (the fraction is probably too large)."""

    @staticmethod
    def frames(camera, far, near=0, per_frame=5):
        """``far`` 40 px people (foot row 340) and ``near`` 300 px people standing closer
        (foot row 700, so the far ones do not borrow their height), spread over frames."""
        people = [still([60 + 180 * k, 300, 75 + 180 * k, 340], colour=k) for k in range(far)]
        people += [still([60 + 180 * (far + k), 400, 160 + 180 * (far + k), 700], colour=far + k) for k in range(near)]
        chunks = [people[i:i + per_frame] for i in range(0, len(people), per_frame)] or [[]]
        return [item(f"{camera}/IMG_{n:04d}.JPG", image(chunk, at(10 * n), n)) for n, chunk in enumerate(chunks)]

    def zone(self, post, camera):
        return post["calibrations"][f"{camera}|Acme TC-1@2000x1200"]["near_zone"]

    def test_ten_people_and_none_near_asks_to_check_the_fraction(self):
        post = experimental_postprocess(self.frames("quiet", far=10))
        self.assertEqual(self.zone(post, "quiet"), {"near_fraction": 0.07, "people_total": 10, "people_near": 0,
                                                    "status": "check_near_fraction"})

    def test_fewer_than_ten_people_or_anyone_near_is_ok(self):
        post = experimental_postprocess(self.frames("nine", far=9) + self.frames("one_near", far=9, near=1)
                           + self.frames("empty", far=0))
        self.assertEqual(self.zone(post, "nine"), {"near_fraction": 0.07, "people_total": 9, "people_near": 0,
                                                   "status": "ok"})
        self.assertEqual(self.zone(post, "one_near"), {"near_fraction": 0.07, "people_total": 10, "people_near": 1,
                                                       "status": "ok"})
        self.assertEqual(self.zone(post, "empty"), {"near_fraction": 0.07, "people_total": 0, "people_near": 0,
                                                    "status": "ok"})

    def test_reports_the_fraction_in_use_and_follows_it(self):
        frames = self.frames("camA", far=9, near=1)
        post = experimental_postprocess(frames, {"near_fraction": 0.5})   # 600 px: nobody is near
        self.assertEqual(self.zone(post, "camA"), {"near_fraction": 0.5, "people_total": 10, "people_near": 0,
                                                   "status": "check_near_fraction"})
        post = experimental_postprocess(self.frames("camA", far=9, near=1), {"near_fraction": 0})
        self.assertEqual(self.zone(post, "camA")["people_near"], 10)

    def test_every_calibration_group_gets_a_diagnostic_and_it_is_json_safe(self):
        post = experimental_postprocess(self.frames("camA", far=3) + self.frames("camB", far=0))
        self.assertEqual(set(post["calibrations"]), {"camA|Acme TC-1@2000x1200", "camB|Acme TC-1@2000x1200"})
        for calibration in post["calibrations"].values():
            self.assertEqual(set(calibration["near_zone"]), {"near_fraction", "people_total", "people_near", "status"})
        json.dumps(post["calibrations"], allow_nan=False)


if __name__ == "__main__":
    unittest.main()
