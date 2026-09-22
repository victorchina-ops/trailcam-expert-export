"""Attribute policy and failure behavior; no model download is needed."""
import math
import tempfile
import unittest
from pathlib import Path

from trailcam.attributes import AttributeEngine, INDEX_NAMES, crop_box, derive_attributes


def sample_scores(**overrides):
    result = {name: 0.01 for name in INDEX_NAMES.values()}
    result.update(age_18_60=0.99, front=0.92, native_female=0.9)
    result.update(overrides)
    return result


class AttributePolicyTests(unittest.TestCase):
    def test_clear_attributes_and_native_scores_are_preserved(self):
        scores = sample_scores(backpack=0.9)
        got = derive_attributes(scores, 200, 400)
        self.assertEqual(got["native_orientation_label"], "front")
        self.assertEqual(got["native_age_label"], "18_60")
        self.assertTrue(got["backpack_presence"])
        self.assertEqual(got["scores"], scores)
        self.assertEqual(got["presentation_proxy"], "feminine_presentation_proxy")

    def test_tiny_person_abstains_even_with_high_native_scores(self):
        scores = sample_scores(backpack=1.0)
        got = derive_attributes(scores, 79, 400)
        self.assertEqual(got["native_orientation_label"], "unknown")
        self.assertEqual(got["native_age_label"], "unknown")
        self.assertIsNone(got["backpack_presence"])
        self.assertEqual(got["presentation_proxy"], "unclear")
        self.assertEqual(got["scores"], scores)

    def test_orientation_conflict_abstains(self):
        got = derive_attributes(sample_scores(front=0.91, back=0.89), 200, 400)
        self.assertEqual(got["native_orientation_label"], "unknown")
        self.assertIn("top_vs_runner_up_margin_below_threshold", got["gate_reasons"]["orientation"])

    def test_ambiguous_bag_and_binary_label_abstain(self):
        got = derive_attributes(sample_scores(backpack=0.5, native_female=0.5), 200, 400)
        self.assertIsNone(got["backpack_presence"])
        self.assertEqual(got["presentation_proxy"], "unclear")

    def test_backpack_boundary_scores_are_inclusive(self):
        self.assertIs(derive_attributes(sample_scores(backpack=0.2), 80, 160)["backpack_presence"], False)
        self.assertIs(derive_attributes(sample_scores(backpack=0.8), 80, 160)["backpack_presence"], True)

    def test_bad_scores_fail_closed(self):
        for invalid in (math.nan, math.inf, -0.01, 1.01):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                derive_attributes(sample_scores(backpack=invalid), 200, 400)
        incomplete = sample_scores()
        del incomplete["front"]
        with self.assertRaises(ValueError):
            derive_attributes(incomplete, 200, 400)

    def test_crop_rounding_clipping_and_invalid_boxes(self):
        self.assertEqual(crop_box([-1.5, 2.2, 101.1, 90.1], 100, 90), [0, 2, 100, 90])
        for invalid in ([9, 1, 3, 10], [0, 0, math.nan, 1], [0, 1, 2], [101, 0, 200, 80]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                crop_box(invalid, 100, 100)

    def test_empty_roster_does_not_open_image(self):
        engine = object.__new__(AttributeEngine)
        engine.device = "cpu"
        engine.requested_device = "auto"
        engine.fallback_reasons = []
        got = engine.analyze(Path("intentionally_missing_image.jpg"), [])
        self.assertEqual(got["persons"], [])
        self.assertEqual(got["status"], "ok")

    def test_bad_person_does_not_drop_roster_or_other_predictions(self):
        from PIL import Image
        engine = object.__new__(AttributeEngine)
        engine.device = "cpu"
        engine.requested_device = "auto"
        engine.fallback_reasons = []
        engine._predict_crop = lambda crop: sample_scores()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            Image.new("RGB", (200, 400)).save(path)
            got = engine.analyze(path, [
                {"person_id": "bad", "bbox_xyxy": [50, 50, 10, 20]},
                {"person_id": "good", "bbox_xyxy": [0, 0, 100, 200]},
            ])
        self.assertEqual(got["status"], "partial_error")
        self.assertEqual(got["people_failed"], 1)
        self.assertEqual([person["person_id"] for person in got["persons"]], ["bad", "good"])
        self.assertEqual(got["persons"][0]["attributes"]["native_orientation_label"], "unknown")
        self.assertEqual(got["persons"][1]["attributes"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
